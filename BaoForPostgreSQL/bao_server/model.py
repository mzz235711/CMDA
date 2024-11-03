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
    loss = [] 
    for i in range(bs):
        mask = ((label_similarity[i] - label_similarity[i].unsqueeze(1)) >= 0).float()
        loss.append(torch.sum(-torch.log(feature_similarity[i] / torch.sum(mask * feature_similarity[i], dim=1))).unsqueeze(0))
    loss = torch.cat(loss)
    loss /= bs
    return loss

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

    targets = torch.tensor(targets)
    return trees, targets

class BaoRegression:
    def __init__(self, verbose=False, have_cache_data=False, alpha=0.25, contrastive=False, meta=False, lr=0.001):
        self.__net = None
        self.__verbose = verbose
        self.__alpha = alpha
        self.__contrastive=contrastive
        self.__meta = meta
        self.__lr = 0.001

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
        path += "_{}_{}_{}".format(self.__alpha, self.__lr, repeat)
        print("Load model from {}".format(path))

        with open(_n_path(path), "rb") as f:
            self.__n = joblib.load(f)
        with open(_channels_path(path), "rb") as f:
            self.__in_channels = joblib.load(f)
            
        self.__net = net.BaoNet(self.__in_channels)
        self.__net.load_state_dict(torch.load(_nn_path(path)))
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

    def fit(self, X, y, train_num=10):
        if isinstance(y, list):
            y = np.array(y)

        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        self.__n = len(X)
            
        # transform the set of trees into feature vectors using a log
        # (assuming the tail behavior exists, TODO investigate
        #  the quantile transformer from scikit)
        y = self.__pipeline.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        
        self.__tree_transform.fit(X)
        X = self.__tree_transform.transform(X)

        X_train = X[:train_num]
        X_test = X[train_num:]
        y_train = y[:train_num]
        y_test = y[train_num:]
        print(len(X_train), len(X_test))

        pairs = list(zip(X_train, y_train))
        dataset = DataLoader(pairs,
                             batch_size=256,
                             shuffle=True,
                             collate_fn=collate)
        
        test_pairs = list(zip(X_test, y_test))
        test_dataset = DataLoader(test_pairs, batch_size=256, shuffle=True, collate_fn=collate)

        # determine the initial number of channels
        for inp, _tar in dataset:
            in_channels = inp[0][0].shape[0]
            break

        self.__log("Initial input channels:", in_channels)

        if self.__have_cache_data:
            assert in_channels == self.__tree_transform.num_operators() + 3
        else:
            assert in_channels == self.__tree_transform.num_operators() + 2

        self.__net = net.BaoNet(in_channels)
        self.__in_channels = in_channels
        if CUDA:
            self.__net = self.__net.cuda()

        optimizer = torch.optim.Adam(self.__net.parameters(), lr=self.__lr)
        loss_fn = torch.nn.MSELoss(reduction='none')
        
        losses = []
        for epoch in range(100):
            loss_accum = 0
            for x, y in dataset:
                if CUDA:
                    y = y.cuda()
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
                    if norm_c != 0:
                        w = w_tilde / norm_c
                    else:
                        w = w_tilde
                y_feature, y_pred = self.__net(x)
                contrastive_loss = ranking_loss(y_feature, y, y_pred)
                est_loss = loss_fn(y_pred, y)
                if self.__contrastive is True:
                    loss = torch.sum(self.__alpha * w * contrastive_loss + (1 - self.__alpha) * w * est_loss)
                else:
                    loss = torch.sum(w * est_loss)
                loss_accum += loss.item()
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_accum /= len(dataset)
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
        pred = self.__net(X)[1].cpu().detach().numpy()
        return self.__pipeline.inverse_transform(pred)

