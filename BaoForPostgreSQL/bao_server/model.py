import json
import numpy as np
import torch
import torch.optim
import joblib
import os
import higher
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from torch.utils.data import DataLoader
import net
from featurize import TreeFeaturizer

CUDA = torch.cuda.is_available()

def _nn_path(base):
    return os.path.join(base, "nn_weights")

def _x_transform_path(base):
    return os.path.join(base, "x_transform")

def _y_transform_path(base):
    return os.path.join(base, "y_transform")

def _channels_path(base):
    return os.path.join(base, "channels")

def _n_path(base):
    return os.path.join(base, "n")


def _inv_log1p(x):
    return np.exp(x) - 1

def ranking_loss(features, true_labels, est_labels, temperature=2):
    #preds = unnormalize_torch(est_labels, min_val, max_val) 
    #targets = unnormalize_torch(true_labels, min_val, max_val) 
    bs = features.shape[0]
    features_norm = features / features.norm(dim=1)[:, None]
    #extend_features = features.unsqueeze(1)
    feature_similarity = torch.mm(features_norm, features_norm.transpose(0, 1))
    feature_similarity = torch.exp(feature_similarity / temperature)
    #label_similarity = 1 / ((true_labels.unsqueeze(1) - true_labels)**2 + 1)
    label_similarity = (true_labels - true_labels.squeeze(1))**2
    # The original denominator for candidate j contains all candidates whose
    # output distance is at least d(i,j). Sorting once per anchor and taking a
    # suffix sum is mathematically equivalent, but avoids the Python anchor
    # loop and its repeated O(batch^2) masks.
    sorted_distances, order = torch.sort(label_similarity, dim=1)
    sorted_similarities = torch.gather(feature_similarity, 1, order)
    if sorted_similarities.is_cuda and torch.are_deterministic_algorithms_enabled():
        # CUDA cumsum has no deterministic implementation.  Multiplication by
        # a lower-triangular matrix computes the same row-wise suffix sums and
        # remains reproducible under CUBLAS_WORKSPACE_CONFIG.
        suffix_matrix = torch.tril(torch.ones(
            bs, bs, dtype=sorted_similarities.dtype,
            device=sorted_similarities.device))
        suffix_similarities = torch.mm(sorted_similarities, suffix_matrix)
    else:
        suffix_similarities = torch.flip(
            torch.cumsum(torch.flip(sorted_similarities, dims=(1,)), dim=1),
            dims=(1,))
    lower_bounds = torch.searchsorted(
        sorted_distances, label_similarity.contiguous(), right=False)
    denominators = torch.gather(suffix_similarities, 1, lower_bounds)
    return (-torch.log(feature_similarity / denominators)).sum(dim=1) / bs


def template_contrastive_loss(features, template_labels, temperature=2.0):
    normalized = torch.nn.functional.normalize(features, dim=1)
    logits = torch.mm(normalized, normalized.transpose(0, 1)) / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
    count = features.shape[0]
    self_mask = ~torch.eye(count, dtype=torch.bool, device=features.device)
    positives = template_labels.view(-1, 1).eq(template_labels.view(1, -1)) & self_mask
    exp_logits = torch.exp(logits) * self_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    positive_count = positives.sum(dim=1)
    losses = (-(log_prob * positives.float()).sum(dim=1) /
              positive_count.clamp(min=1).float())
    return torch.where(positive_count > 0, losses, torch.zeros_like(losses))


def _broadcast_weighted_objective(weights, representation_loss,
                                  regression_loss, alpha):
    """Evaluate the historical broadcasted objective without a B x B tensor.

    ``representation_loss`` and ``weights`` have shape [B], while Bao's
    unreduced regression loss has shape [B, 1].  The original expression
    therefore broadcasts to [B, B] before summation.  Expanding that sum
    algebraically preserves both its value and gradients while avoiding the
    quadratic temporary and autograd graph.
    """
    batch_size = regression_loss.shape[0]
    regression_sum = regression_loss.sum()
    if not torch.is_tensor(weights) or weights.ndim == 0:
        return batch_size * weights * (
            alpha * representation_loss.sum()
            + (1 - alpha) * regression_sum)
    return (alpha * batch_size * torch.sum(weights * representation_loss)
            + (1 - alpha) * torch.sum(weights) * regression_sum)

'''
def ranking_loss(features, true_labels, est_labels, temperature=2):
    features_norm = features / features.norm(dim=1)[:, None]
    #extend_features = features.unsqueeze(1)
    feature_similarity = torch.mm(features_norm, features_norm.transpose(0, 1))
    feature_similarity = torch.exp(feature_similarity / temperature)
    label_similarity = 1 / ((true_labels.unsqueeze(1) - est_labels)**2 + 1)
    return -torch.log(torch.sum(label_similarity * feature_similarity, dim=1) / torch.sum(feature_similarity, dim=1))
'''

class BaoData:
    def __init__(self, data):
        assert data
        self.__data = data

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return (self.__data[idx]["tree"],
                self.__data[idx]["target"])

def collate(x):
    trees = []
    targets = []

    for tree, target in x:
        trees.append(tree)
        targets.append(target)

    targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    return trees, targets


def collate_template(x):
    trees, targets, template_labels = zip(*x)
    return (list(trees),
            torch.from_numpy(np.asarray(targets, dtype=np.float32)),
            torch.tensor(template_labels, dtype=torch.long))


def collate_weighted(x):
    trees, targets, weights = zip(*x)
    return (list(trees),
            torch.from_numpy(np.asarray(targets, dtype=np.float32)),
            torch.tensor(weights, dtype=torch.float32))

class BaoRegression:
    def __init__(self, verbose=False, have_cache_data=False, alpha=0.25, contrastive=False, meta=False, lr=0.001,
                 template_contrastive=False):
        self.__net = None
        self.__verbose = verbose
        self.__alpha = alpha
        self.__contrastive=contrastive
        self.__meta = meta
        self.__template_contrastive = template_contrastive
        self.__lr = lr

        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p,
            validate=True)
        scale_transformer = preprocessing.MinMaxScaler()

        self.__pipeline = Pipeline([("log", log_transformer),
                                    ("scale", scale_transformer)])
        
        self.__tree_transform = TreeFeaturizer()
        self.__have_cache_data = have_cache_data
        self.__in_channels = None
        self.__n = 0
        
    def __log(self, *args):
        if self.__verbose:
            print(*args)

    def num_items_trained_on(self):
        return self.__n
            
    def load(self, path, repeat):
        if self.__contrastive is True:
            path += "_contrastive"
        if self.__meta is True:
            path += "_meta"
        if self.__template_contrastive:
            path += "_template"
        path += "_{}_{}_{}".format(self.__alpha, self.__lr, repeat)
        print("Load model from {}".format(path))

        with open(_n_path(path), "rb") as f:
            self.__n = joblib.load(f)
        with open(_channels_path(path), "rb") as f:
            self.__in_channels = joblib.load(f)
            
        self.__net = net.BaoNet(self.__in_channels)
        self.__net.load_state_dict(torch.load(
            _nn_path(path), map_location="cuda" if CUDA else "cpu"))
        self.__net.eval()
        
        with open(_y_transform_path(path), "rb") as f:
            self.__pipeline = joblib.load(f)
        with open(_x_transform_path(path), "rb") as f:
            self.__tree_transform = joblib.load(f)

    def save(self, path, repeat):
        # try to create a directory here
        if self.__contrastive is True:
            path += "_contrastive"
        if self.__meta is True:
            path += "_meta"
        if self.__template_contrastive:
            path += "_template"
        path += "_{}_{}_{}".format(self.__alpha, self.__lr, repeat)
        os.makedirs(path, exist_ok=True)
        print("Save model to {}".format(path))
        
        torch.save(self.__net.state_dict(), _nn_path(path))
        with open(_y_transform_path(path), "wb") as f:
            joblib.dump(self.__pipeline, f)
        with open(_x_transform_path(path), "wb") as f:
            joblib.dump(self.__tree_transform, f)
        with open(_channels_path(path), "wb") as f:
            joblib.dump(self.__in_channels, f)
        with open(_n_path(path), "wb") as f:
            joblib.dump(self.__n, f)

    def fit(self, X, y, train_num=10, epochs=100, batch_size=256, seed=0,
            template_labels=None, sample_weights=None, warm_start=False):
        if isinstance(y, list):
            y = np.array(y)

        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        self.__n = len(X)
            
        # transform the set of trees into feature vectors using a log
        # (assuming the tail behavior exists, TODO investigate
        #  the quantile transformer from scikit)
        if warm_start:
            if self.__net is None:
                raise ValueError("warm_start requires a loaded Bao model")
            y = self.__pipeline.transform(y.reshape(-1, 1)).astype(np.float32)
        else:
            y = self.__pipeline.fit_transform(y.reshape(-1, 1)).astype(np.float32)
            self.__tree_transform.fit(X)
        X = self.__tree_transform.transform(X)

        X_train = X[:train_num]
        X_test = X[train_num:]
        y_train = y[:train_num]
        y_test = y[train_num:]
        print(len(X_train), len(X_test))

        if sample_weights is not None:
            if len(sample_weights) != len(X):
                raise ValueError("Sample weights must align with all examples")
            pairs = list(zip(X_train, y_train, sample_weights[:train_num]))
        elif self.__template_contrastive:
            if template_labels is None or len(template_labels) != len(X):
                raise ValueError("Template labels must align with all training examples")
            pairs = list(zip(X_train, y_train, template_labels[:train_num]))
        else:
            pairs = list(zip(X_train, y_train))
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataset = DataLoader(pairs,
                             batch_size=batch_size,
                             shuffle=True,
                             collate_fn=(collate_weighted if sample_weights is not None
                                         else collate_template if self.__template_contrastive
                                         else collate),
                             generator=generator)
        
        test_pairs = list(zip(X_test, y_test))
        test_generator = torch.Generator()
        test_generator.manual_seed(seed + 1)
        test_dataset = DataLoader(test_pairs, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate, generator=test_generator)

        # determine the initial number of channels
        for initial_batch in dataset:
            inp = initial_batch[0]
            in_channels = inp[0][0].shape[0]
            break

        self.__log("Initial input channels:", in_channels)

        if self.__have_cache_data:
            assert in_channels == self.__tree_transform.num_operators() + 3
        else:
            assert in_channels == self.__tree_transform.num_operators() + 2

        if warm_start:
            if in_channels != self.__in_channels:
                raise ValueError(
                    "Warm-start feature width changed: {} != {}".format(
                        in_channels, self.__in_channels))
            self.__net.train()
        else:
            self.__net = net.BaoNet(in_channels)
            self.__in_channels = in_channels
            if CUDA:
                self.__net = self.__net.cuda()

        optimizer = torch.optim.Adam(self.__net.parameters(), lr=self.__lr)
        loss_fn = torch.nn.MSELoss(reduction='none')
        
        losses = []
        for epoch in range(epochs):
            loss_accum = torch.zeros((), device="cuda" if CUDA else "cpu")
            for batch in dataset:
                if sample_weights is not None:
                    x, y, batch_weights = batch
                elif self.__template_contrastive:
                    x, y, batch_template_labels = batch
                else:
                    x, y = batch
                if CUDA:
                    y = y.cuda()
                    if sample_weights is not None:
                        batch_weights = batch_weights.cuda()
                    if self.__template_contrastive:
                        batch_template_labels = batch_template_labels.cuda()
                if sample_weights is not None:
                    w = batch_weights / batch_weights.sum()
                else:
                    w = 1 / len(x)
                if self.__meta is True:
                    interval = 0
                else:
                    interval = 10
                if epoch % 10 == interval:
                    with higher.innerloop_ctx(self.__net, optimizer) as (meta_model, meta_optimizer):
                        y_feature, y_pred = meta_model(x)
                        est_loss = loss_fn(y_pred, y)
                        eps = torch.zeros(est_loss.size(), requires_grad=True, device=y.device)
                        meta_loss = torch.sum(est_loss * eps)
                        meta_optimizer.step(meta_loss)

                        l_g_meta = []
                        for x_test, y_test in test_dataset:
                            if CUDA:
                                y_test = y_test.cuda()
                            test_y_feature, test_y_pred = meta_model(x_test)
                            l_g_meta.append(loss_fn(test_y_pred, y_test))
                        l_g_meta = torch.mean(torch.cat(l_g_meta))
                        grad_eps = torch.autograd.grad(l_g_meta, eps)[0].detach()
                    
                    w_tilde = torch.clamp(-grad_eps, min=0)
                    norm_c = torch.sum(w_tilde)
                    # w_tilde is nonnegative; clamping only changes the zero
                    # denominator case, where the original branch also
                    # returned the all-zero vector.
                    w = w_tilde / norm_c.clamp_min(torch.finfo(w_tilde.dtype).eps)
                y_feature, y_pred = self.__net(x)
                est_loss = loss_fn(y_pred, y)
                if sample_weights is not None:
                    loss = torch.sum(w * est_loss.squeeze(1))
                elif self.__template_contrastive:
                    contrastive_loss = template_contrastive_loss(
                        y_feature, batch_template_labels)
                    loss = torch.sum(self.__alpha * w * contrastive_loss + (1 - self.__alpha) * w * est_loss.squeeze(1))
                elif self.__contrastive is True:
                    contrastive_loss = ranking_loss(y_feature, y, y_pred)
                    loss = _broadcast_weighted_objective(
                        w, contrastive_loss, est_loss, self.__alpha)
                else:
                    loss = torch.sum(w * est_loss)
                loss_accum = loss_accum + loss.detach()
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_accum = (loss_accum / len(dataset)).item()
            losses.append(loss_accum)
            if epoch % 5 == 0:
                self.__log("Epoch", epoch, "training loss:", loss_accum)

            # stopping condition
            if len(losses) > 10 and losses[-1] < 0.1:
                last_two = np.min(losses[-2:])
                if last_two > losses[-10] or (losses[-10] - last_two < 0.0001):
                    self.__log("Stopped training from convergence condition at epoch", epoch)
                    break
        else:
            self.__log("Stopped training after max epochs")

    def predict(self, X):
        if not isinstance(X, list):
            X = [X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]

        X = self.__tree_transform.transform(X)
        
        self.__net.eval()
        with torch.inference_mode():
            pred = self.__net(X)[1].cpu().numpy()
        return self.__pipeline.inverse_transform(pred)

    def representation_features(self, X):
        """Return the learned Bao representation used for CBP clustering."""
        if not isinstance(X, list):
            X = [X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        transformed = self.__tree_transform.transform(X)
        self.__net.eval()
        with torch.no_grad():
            features, _ = self.__net(transformed)
        return features.cpu().numpy()
