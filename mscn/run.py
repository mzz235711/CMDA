import argparse
import time
import os
import pickle
import json
import random
import hashlib
import psycopg2
import higher
try:
    import nni
except Exception:
    class _NniFallback:
        @staticmethod
        def get_next_parameter():
            return {}

        @staticmethod
        def report_final_result(_result):
            return None

    nni = _NniFallback()

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data import *
from util import *
from model import *
from dataset import *
#from data_augementation import *


def set_global_seed(seed):
    """Seed all random generators used by the MSCN experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_queries(pool_file):
    with open(pool_file) as pool_stream:
        return sum(1 for _ in pool_stream)


def build_adaptation_indices(candidate_indices, budget, seed):
    if budget <= 0:
        raise ValueError("adaptation_budget must be positive")
    if budget > len(candidate_indices):
        raise ValueError("adaptation_budget exceeds the target candidate pool")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(candidate_indices, size=budget, replace=False))
    return indices.tolist()


def indices_hash(indices):
    encoded = ",".join(str(i) for i in indices).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_index_list(value):
    return None if value is None else [int(item) for item in value.split(",") if item]


def load_or_create_target_split(output_root, dataset, pool_file,
                                candidate_fraction, split_seed):
    if not 0 < candidate_fraction < 1:
        raise ValueError("candidate_fraction must be between 0 and 1")
    pool_size = count_queries(pool_file)
    candidate_size = int(round(pool_size * candidate_fraction))
    candidate_size = min(max(candidate_size, 1), pool_size - 1)
    rng = np.random.default_rng(split_seed)
    permutation = rng.permutation(pool_size)
    candidate_indices = np.sort(permutation[:candidate_size]).tolist()
    evaluation_indices = np.sort(permutation[candidate_size:]).tolist()
    split_dir = os.path.join(output_root, dataset, "target_splits")
    os.makedirs(split_dir, exist_ok=True)
    fraction_label = "{:.4f}".format(candidate_fraction).replace(".", "p")
    split_file = os.path.join(
        split_dir, "split_seed_{}_candidate_{}.json".format(
            split_seed, fraction_label))
    expected = {
        "dataset": dataset,
        "target_pool_file": os.path.abspath(pool_file),
        "target_pool_size": pool_size,
        "split_seed": split_seed,
        "candidate_fraction": candidate_fraction,
        "candidate_indices": candidate_indices,
        "evaluation_indices": evaluation_indices,
        "candidate_id": indices_hash(candidate_indices),
        "evaluation_id": indices_hash(evaluation_indices),
    }
    expected["split_id"] = hashlib.sha256(
        (expected["candidate_id"] + expected["evaluation_id"]).encode("utf-8")
    ).hexdigest()
    if os.path.exists(split_file):
        with open(split_file) as split_stream:
            existing = json.load(split_stream)
        if existing != expected:
            raise RuntimeError("Existing target split does not match: {}".format(split_file))
        return existing, split_file
    with open(split_file, "w") as split_stream:
        json.dump(expected, split_stream, indent=2, sort_keys=True)
    return expected, split_file


def resolve_method_name(explicit_name, contrastive, meta):
    if explicit_name:
        return explicit_name
    if contrastive and meta:
        return "cmda"
    if contrastive:
        return "cmda_c"
    if meta:
        return "cmda_m"
    return "adapt"


def load_or_create_selection_manifest(output_root, dataset, budget, seed,
                                      pool_file, split, split_file):
    pool_size = split["target_pool_size"]
    generated_indices = build_adaptation_indices(
        split["candidate_indices"], budget, seed)
    selection_dir = os.path.join(
        output_root, dataset, "budget_{}".format(budget), "selections")
    os.makedirs(selection_dir, exist_ok=True)
    selection_file = os.path.join(
        selection_dir, "seed_{:03d}.json".format(seed))
    expected = {
        "dataset": dataset,
        "seed": seed,
        "adaptation_budget": budget,
        "adaptation_pool_file": os.path.abspath(pool_file),
        "adaptation_pool_size": pool_size,
        "adaptation_indices": generated_indices,
        "candidate_pool_size": len(split["candidate_indices"]),
        "split_id": split["split_id"],
        "split_manifest": os.path.abspath(split_file),
    }
    expected["selection_id"] = indices_hash(generated_indices)
    if os.path.exists(selection_file):
        with open(selection_file) as selection_stream:
            existing = json.load(selection_stream)
        if existing != expected:
            raise RuntimeError(
                "Existing adaptation selection does not match the requested "
                "dataset/budget/seed: {}".format(selection_file))
        return existing, selection_file
    with open(selection_file, "w") as selection_stream:
        json.dump(expected, selection_stream, indent=2, sort_keys=True)
    return expected, selection_file

    

def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)

def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))
    return qerror

def qerror_loss(preds, targets, min_val, max_val):
    preds = unnormalize_torch(preds, min_val, max_val).reshape(-1)
    targets = unnormalize_torch(targets, min_val, max_val).reshape(-1)
    # Keep the original per-tuple definition while avoiding one GPU-to-CPU
    # synchronization per tuple.
    return torch.log(torch.maximum(preds / targets, targets / preds))

def ranking_loss(features, true_labels, est_labels, min_val, max_val, temperature=2):
    #preds = unnormalize_torch(est_labels, min_val, max_val) 
    #targets = unnormalize_torch(true_labels, min_val, max_val) 
    bs = features.shape[0]
    # Numerically stable cosine normalization. This is identical to the
    # original division for nonzero embeddings and prevents a zero ReLU
    # embedding from turning the full similarity matrix into NaNs.
    features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
    #extend_features = features.unsqueeze(1)
    feature_similarity = torch.mm(features_norm, features_norm.transpose(0, 1))
    feature_similarity = torch.exp(feature_similarity / temperature)
    #label_similarity = 1 / ((true_labels.unsqueeze(1) - true_labels)**2 + 1)
    label_similarity = (true_labels.unsqueeze(1) - true_labels)**2
    # For candidate j, the original denominator sums similarities to all k
    # whose output distance is no smaller than j's. Sorting each anchor's
    # distances and taking suffix sums computes exactly the same denominator,
    # without the Python anchor loop or an O(batch^3) comparison tensor.
    sorted_distances, order = torch.sort(label_similarity, dim=1)
    sorted_similarities = torch.gather(feature_similarity, 1, order)
    suffix_similarities = torch.flip(
        torch.cumsum(torch.flip(sorted_similarities, dims=(1,)), dim=1),
        dims=(1,))
    lower_bounds = torch.searchsorted(
        sorted_distances, label_similarity.contiguous(), right=False)
    denominators = torch.gather(suffix_similarities, 1, lower_bounds)
    return (-torch.log(feature_similarity / denominators)).sum(dim=1) / bs


def template_contrastive_loss(features, template_labels, temperature=2.0):
    """Per-anchor supervised contrastive loss using query-template identity."""
    normalized = torch.nn.functional.normalize(features, dim=1)
    logits = torch.mm(normalized, normalized.transpose(0, 1)) / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
    count = features.shape[0]
    self_mask = ~torch.eye(count, dtype=torch.bool, device=features.device)
    positive_mask = template_labels.view(-1, 1).eq(
        template_labels.view(1, -1)) & self_mask
    exp_logits = torch.exp(logits) * self_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    positive_count = positive_mask.sum(dim=1)
    losses = (-(log_prob * positive_mask.float()).sum(dim=1) /
              positive_count.clamp(min=1).float())
    return torch.where(positive_count > 0, losses, torch.zeros_like(losses))
    #return -torch.log(torch.sum(label_similarity * feature_similarity, dim=1) / torch.sum(feature_similarity, dim=1))
    #loss = 0.0
    #for i in range(bs):
    #    positive_pair = feature_similarity[i]
    #    label_difference = label_similarity[i] > label_similarity[i].unsqueeze(1)
    #    negative_pairs = 
    #return similarity_sum / feature_similarity_sum

def csa_loss(x, y, class_eq):
    margin = 1
    dist = F.pairwise_distance(x, y)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()

#def triplet_loss(anchor, pos, neg):

def predict_triplet_query(model, data_loader, cuda, adaptation=False):
    preds = []
    t_total = 0.

    model.eval()
    with torch.inference_mode():
        for batch_idx, data_batch in enumerate(data_loader):

        #anchor_samples, anchor_cols, anchor_opvals, anchor_joins, anchor_sample_masks, \
        #anchor_col_masks, anchor_opval_masks, anchor_join_masks, anchor_labels, anchor_template_labels, \
        #positive_samples, positive_cols, positive_opvals, positive_joins, positive_sample_masks, \
        #positive_col_masks, positive_opval_masks, positive_join_masks, positive_labels, positive_template_labels, \
        #negative_samples, negative_cols, negative_opvals, negative_joins, negative_sample_masks, \
        #negative_col_masks, negative_opval_masks, negative_join_masks, negative_labels, negative_template_labels = data_batch
            samples, predicates, joins, sample_masks, \
                predicate_masks, join_masks = data_batch[:6]

            if cuda == 'cuda':
                samples, predicates, joins = samples.cuda(), predicates.cuda(), joins.cuda()
                sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        #samples, predicates, joins = Variable(samples), Variable(predicates), Variable(joins)
        #sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(
        #    join_masks)

            t = time.time()
        #outputs, _ = model(samples, cols, opvals, joins, sample_masks, col_masks, opval_masks, join_masks)
            outputs, _ = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            t_total += time.time() - t

            preds.extend(outputs.unbind(0))

    return preds, t_total

def predict_query(model, data_loader, cuda, adaptation=False):
    preds = []
    t_total = 0.

    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):

        if adaptation is True:
            #_, _, _, _, _, _, _, _, _, _, \
            #    samples, cols, opvals, joins, targets, sample_masks, col_masks, opval_masks, join_masks, _ = data_batch
            _, _, _, _, _, _, _, _, _, _, \
                samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks, _ = data_batch
        else:
            #samples, cols, opvals, joins, targets, sample_masks, col_masks, opval_masks, join_masks = data_batch
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch

        #print(samples.size())
        #print(cols.size())
        #print(opvals.size())
        #print(joins.size())
        #print(sample_masks.size())
        #print(col_masks.size())
        #print(opval_masks.size())
        #print(join_masks.size())
        #print(targets.size())
        if cuda == 'cuda':
            #samples, cols, opvals, joins, targets = samples.cuda(), cols.cuda(), opvals.cuda(), joins.cuda(), targets.cuda()
            #sample_masks, col_masks, opval_masks, join_masks = sample_masks.cuda(), col_masks.cuda(), opval_masks.cuda(), join_masks.cuda()
            samples, predicates, joins, targets = samples.cuda(), predicates.cuda(), joins.cuda(), targets.cuda()
            sample_masks, predicate_masks, join_masks = sample_masks.cuda(), predicate_masks.cuda(), join_masks.cuda()
        #samples, cols, opvals, joins, targets = Variable(samples), Variable(cols), Variable(opvals), Variable(joins), Variable(targets)
        #sample_masks, col_masks, opval_masks, join_masks = Variable(sample_masks), Variable(col_masks), Variable(opval_masks), Variable(join_masks)
        samples, predicates, joins, targets = Variable(samples), Variable(predicates), Variable(joins), Variable(targets)
        sample_masks, predicate_masks, join_masks = Variable(sample_masks), Variable(predicate_masks), Variable(join_masks)

        t = time.time()
        #outputs, _ = model(samples, cols, opvals, joins, sample_masks, col_masks, opval_masks, join_masks)
        with torch.inference_mode():
            outputs, _ = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
        t_total += time.time() - t

        preds.extend(outputs.unbind(0))

    return preds, t_total


def predict(dataset, predict_file, result_file, save_folder, bitmap_file, cuda='cuda', batch_size=1024, hidden_units=256, num_samples=0, adaptation=True, alpha=0.25, query_num=0, directly_adapt=False, contrastive=True, meta=False, exclude_indices=None, include_indices=None):


    # Load other structures
    load_dicts = []
    structure_file = save_folder + '/{}_structure'.format(dataset)
    if adaptation is True:
        structure_file += "_adaptation"
    if contrastive is True:
        structure_file += "_contrastive"
    if meta is True:
        structure_file += "_meta"
    structure_file += ".pkl"
    with open(structure_file, 'rb') as fp:
        load_dicts = pickle.load(fp)
    column_min_max_vals = load_dicts['column_min_max_vals']
    table2vec = load_dicts['table2vec']
    join2vec = load_dicts['join2vec']
    column2vec = load_dicts['column2vec']
    op2vec = load_dicts['op2vec']
    vec2table = load_dicts['vec2table']
    vec2join = load_dicts['vec2join']
    vec2column = load_dicts['vec2column']
    vec2op = load_dicts['vec2op']
    min_val = load_dicts['min_val']
    max_val = load_dicts['max_val']
    max_num_predicates = load_dicts['max_predicates']
    max_num_joins = load_dicts['max_joins']
    templates = load_dicts['templates']
    template_distribution = load_dicts['template_distribution']
    idxdicts = [vec2table, vec2join, vec2column, vec2op]

    sample_feats = len(table2vec) + num_samples
    col_feats = len(column2vec)
    opval_feats = len(op2vec) + 1
    join_feats = len(join2vec)
    # Load the model
    if adaptation is True:
        if directly_adapt:
            model_file = save_folder + '/{}_directly_adaptation_metamodel_{}_{}.pkl'.format(dataset, alpha, query_num)
        else:
            model_file = save_folder + '/{}_adaptation_model_{}_{}'.format(dataset, alpha, query_num)
            if contrastive is True:
                model_file += "_contrastive"
            if meta is True:
                model_file += "_meta"
            model_file += '.pkl'
    else:
        model_file = save_folder + '/{}_model_{}'.format(dataset, alpha)
        if contrastive is True:
            model_file += "_contrastive"
        if meta is True:
            model_file += "_meta"
        model_file += '.pkl'
    print("Load model pkl from {}".format(model_file))
    model = SetConv(sample_feats, col_feats, opval_feats, join_feats, hidden_units)
    model.load_state_dict(torch.load(model_file))
    if cuda == 'cuda':
        model.cuda()
    # Load test data
    #file_name = "workloads/" + workload_name
    _, _, tables, joins, predicates, label, template_labels = load_query(predict_file, templates)
    samples = load_samples(bitmap_file, len(tables), num_samples)
    if include_indices is not None:
        keep = list(include_indices)
        tables = [tables[i] for i in keep]
        joins = [joins[i] for i in keep]
        predicates = [predicates[i] for i in keep]
        label = [label[i] for i in keep]
        template_labels = [template_labels[i] for i in keep]
        samples = [samples[i] for i in keep]
    elif exclude_indices:
        excluded = set(exclude_indices)
        keep = [i for i in range(len(tables)) if i not in excluded]
        tables = [tables[i] for i in keep]
        joins = [joins[i] for i in keep]
        predicates = [predicates[i] for i in keep]
        label = [label[i] for i in keep]
        template_labels = [template_labels[i] for i in keep]
        samples = [samples[i] for i in keep]

    # Get feature encoding and proper normalization
    #samples_test= []
    samples_test = encode_samples(tables, samples, table2vec)
    #cols_test, opvals_test, joins_test = encode_col_opval_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    predicatess_test, joins_test = encode_predicate_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    labels_test, _, _ = normalize_labels(label, min_val, max_val)

    print("Number of test samples: {}".format(len(labels_test)))

    #max_num_cols = max([len(p) for p in cols_test])
    #max_num_opvals = max([len(p) for p in opvals_test])
    #max_num_joins = max([len(j) for j in joins_test])

    # Get test set predictions
    #test_data = make_dataset(samples_test, cols_test, opvals_test, joins_test, labels_test, max_num_joins, max_num_cols, max_num_opvals, template_labels)
    test_data = make_dataset(samples_test, predicatess_test, joins_test, labels_test, max_num_joins, max_num_predicates, template_labels)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    #preds_test, t_total = predict(model, test_data_loader, cuda)
    

    preds, t_total = predict_query(model, test_data_loader, cuda) 

    print("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_test_unnorm = unnormalize_labels(preds, min_val, max_val, cuda)

    # Print metrics
    print("\nQ-Error " + dataset + ":")
    qerror = print_qerror(preds_test_unnorm, label)
    nni.report_final_result(np.median(qerror))

    metrics = {
        "num_evaluation_queries": len(qerror),
        "mean_qerror": float(np.mean(qerror)),
        "median_qerror": float(np.median(qerror)),
        "p90_qerror": float(np.percentile(qerror, 90)),
        "p95_qerror": float(np.percentile(qerror, 95)),
        "p99_qerror": float(np.percentile(qerror, 99)),
        "max_qerror": float(np.max(qerror)),
        "prediction_time_per_query_ms": float(t_total / len(labels_test) * 1000),
    }
    metrics_file = os.path.join(os.path.dirname(result_file), "metrics.json")
    with open(metrics_file, "w") as metrics_stream:
        json.dump(metrics, metrics_stream, indent=2, sort_keys=True)

    # Write predictions
    #os.makedirs(os.path.dirname(result_file), exist_ok=True)
    '''
    if adaptation is True:
        if directly_adapt:
            result_file = save_folder + '/{}_directly_adaptation_metamodel_{}_{}.csv'.format(dataset, alpha, query_num)
        else:
            result_file = save_folder + '/{}_adaptation_metamodel_{}_{}_result.csv'.format(dataset, alpha, query_num)
    else:
        result_file = save_folder + '/{}_metamodel_{}_result.csv'.format(dataset, alpha)
    '''
    with open(result_file, "w") as f:
        for i in range(len(preds_test_unnorm)):
            f.write(str(preds_test_unnorm[i]) + "," + str(label[i]) + "," + str(qerror[i]) + "\n")


def extract_representation_features(dataset, query_file, bitmap_file,
                                    save_folder, indices, cuda='cpu',
                                    batch_size=512, hidden_units=256,
                                    num_samples=1000, alpha=0.8,
                                    contrastive=False, meta=False):
    """Extract MSCN embeddings for ShiftHandler CBP selection."""
    structure_file = os.path.join(save_folder, "{}_structure".format(dataset))
    if contrastive:
        structure_file += "_contrastive"
    if meta:
        structure_file += "_meta"
    structure_file += ".pkl"
    with open(structure_file, "rb") as stream:
        saved = pickle.load(stream)
    table2vec = saved["table2vec"]
    join2vec = saved["join2vec"]
    column2vec = saved["column2vec"]
    op2vec = saved["op2vec"]
    model = SetConv(len(table2vec) + num_samples, len(column2vec),
                    len(op2vec) + 1, len(join2vec), hidden_units)
    model_file = os.path.join(
        save_folder, "{}_model_{}{}{}.pkl".format(
            dataset, alpha, "_contrastive" if contrastive else "",
            "_meta" if meta else ""))
    model.load_state_dict(torch.load(
        model_file, map_location=torch.device(cuda)))
    if cuda == "cuda":
        model.cuda()
    _, _, tables, joins, predicates, labels, template_labels = load_query(
        query_file, saved["templates"])
    samples = load_samples(bitmap_file, len(tables), num_samples)
    keep = list(indices)
    tables = [tables[i] for i in keep]
    joins = [joins[i] for i in keep]
    predicates = [predicates[i] for i in keep]
    labels = [labels[i] for i in keep]
    template_labels = [template_labels[i] for i in keep]
    samples = [samples[i] for i in keep]
    encoded_samples = encode_samples(tables, samples, table2vec)
    encoded_predicates, encoded_joins = encode_predicate_data(
        predicates, joins, saved["column_min_max_vals"], column2vec,
        op2vec, join2vec)
    normalized_labels, _, _ = normalize_labels(
        labels, saved["min_val"], saved["max_val"])
    data = make_dataset(
        encoded_samples, encoded_predicates, encoded_joins,
        normalized_labels, saved["max_joins"], saved["max_predicates"],
        template_labels)
    loader = DataLoader(data, batch_size=batch_size)
    features = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            sample, predicate, join = batch[0], batch[1], batch[2]
            # make_dataset stores the regression target at position 3.
            sample_mask, predicate_mask, join_mask = (
                batch[4], batch[5], batch[6])
            if cuda == "cuda":
                sample, predicate, join = sample.cuda(), predicate.cuda(), join.cuda()
                sample_mask = sample_mask.cuda()
                predicate_mask = predicate_mask.cuda()
                join_mask = join_mask.cuda()
            _, embedding = model(sample, predicate, join, sample_mask,
                                 predicate_mask, join_mask)
            features.append(embedding.cpu().numpy())
    return np.vstack(features), labels


def adaptation_triplet(dataset, train_query_file, adaptation_query_file, save_folder, train_bitmap_file, adaptation_bitmap_file, conn, cursor, cuda='cuda', batch_size=1024, hidden_units=256, num_epochs=100, alpha=0.25, num_samples=0, query_num=0, directly_adapt=False, contrastive=True, meta=False):
    structure_file = save_folder + '/{}_structure'.format(dataset) 
    if contrastive is True:
        structure_file += "_contrastive"
    if meta is True:
        structure_file += "_meta"
    structure_file += ".pkl"
    with open(structure_file, 'rb') as fp:
        load_dicts = pickle.load(fp)
        column_min_max_vals = load_dicts['column_min_max_vals']
        min_val = load_dicts['min_val']
        max_val = load_dicts['max_val']
        table2vec = load_dicts['table2vec']
        join2vec = load_dicts['join2vec']
        column2vec = load_dicts['column2vec']
        op2vec = load_dicts['op2vec']
        vec2table = load_dicts['vec2table']
        vec2join = load_dicts['vec2join']
        vec2column = load_dicts['vec2column']
        vec2op = load_dicts['vec2op']
        max_num_joins = load_dicts['max_joins']
        #max_num_cols = load_dicts['max_cols']
        #max_num_opvals = load_dicts['max_opvals']
        max_num_predicates = load_dicts['max_predicates']
        templates = load_dicts['templates']
        template_distribution = load_dicts['template_distribution']

    idxdicts = [vec2table, vec2column, vec2op, vec2join]
    #statistics_max = [max_num_cols, max_num_opvals, max_num_joins]
    statistics_max = [max_num_predicates, max_num_joins]
    sample_feats = len(table2vec) + num_samples
    col_feats = len(column2vec)
    opval_feats = len(op2vec) + 1
    join_feats = len(join2vec)

    model_file = save_folder + '/{}_model_{}'.format(dataset, alpha)
    if contrastive is True:
        model_file += "_contrastive"
    if meta is True:
        model_file += "_meta"
    model_file += '.pkl'
    model = SetConv(sample_feats, col_feats, opval_feats, join_feats, hidden_units)
    if directly_adapt is False:
        model.load_state_dict(torch.load(model_file))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if cuda == 'cuda':
        model.cuda()
    #model.sample_mlp1.requires_grad = False
    #model.sample_mlp2.requires_grad = False
    #model.predicate_mlp1.requires_grad = False
    #model.predicate_mlp2.requires_grad = False
    #model.join_mlp1.requires_grad = False
    #model.join_mlp2.requires_grad = False
    #model.part_mlp1.requires_grad = False
    #model.part_mlp2.requires_grad = False
    #anchor_train_tensors, positive_train_tensors, negative_train_tensors, anchor_train_labels, positive_train_labels, negative_train_labels, \
    #    anchor_train_template_labels, positive_train_template_labels, negative_train_template_labels, \
    #    anchor_test_tensors, positive_test_tensors, negative_test_tensors, anchor_test_labels, positive_test_labels, negative_test_labels, \
    #    anchor_test_template_labels, positive_test_template_labels, negative_test_template_labels, train_indices, test_indices, train_large_differnce, test_large_difference =  get_generated_source_target_datasets(train_query_file, adaptation_query_file, train_bitmap_file, adaptation_bitmap_file, table2vec, join2vec, column2vec, op2vec, \
    #                                     idxdicts, column_min_max_vals, min_val, max_val, max_num_joins, max_num_cols, max_num_opvals, templates, conn, cursor, model, num_samples, query_num, cuda)
    anchor_train_tensors, positive_train_tensors, negative_train_tensors, anchor_train_labels, positive_train_labels, negative_train_labels, \
        anchor_train_template_labels, positive_train_template_labels, negative_train_template_labels, \
        anchor_test_tensors, positive_test_tensors, negative_test_tensors, anchor_test_labels, positive_test_labels, negative_test_labels, \
        anchor_test_template_labels, positive_test_template_labels, negative_test_template_labels, train_indices, test_indices, train_large_differnce, test_large_difference \
            =  get_generated_source_target_datasets(train_query_file, adaptation_query_file, train_bitmap_file, adaptation_bitmap_file, table2vec, join2vec, column2vec, op2vec, \
                                         idxdicts, column_min_max_vals, min_val, max_val, max_num_joins, max_num_predicates, templates, conn, cursor, model, num_samples, query_num, cuda)


    tmpfp = open('{}_weight.txt'.format(dataset), 'w')
    test_dataset, negative_test_labels_tensor = rearrange_negative_data_with_template(anchor_test_labels, positive_test_labels, negative_test_labels, anchor_test_template_labels, positive_test_template_labels, negative_test_template_labels, anchor_test_tensors, positive_test_tensors, negative_test_tensors, test_indices, test_large_difference)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
    train_dataset, negative_train_labels_tensor = rearrange_negative_data_with_template(anchor_train_labels, positive_train_labels, negative_train_labels, anchor_train_template_labels, positive_train_template_labels, negative_train_template_labels, anchor_train_tensors, positive_train_tensors, negative_train_tensors, train_indices, train_large_differnce)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    #if contrastive is True:
    #    for param in model.parameters():
    #        param.requires_grad = False
    #    for param in model.out_mlp1.parameters():
    #        param.requires_grad = True
    #    for param in model.out_mlp2.parameters():
    #        param.requires_grad = True
    #    model.out_mlp1.requires_grad = True 
    #    model.out_mlp2.requires_grad = True
    for epoch in range(num_epochs):
        #meta_model = SetConv(sample_feats, col_feats, opval_feats, join_feats, hidden_units)
        #meta_model.load_state_dict(model.state_dict())
        #if cuda == 'cuda':
        #    meta_model.cuda()
        loss_total = 0.0
        for batch_idx, data_batch in enumerate(train_data_loader):
            #anchor_samples, anchor_cols, anchor_opvals, anchor_joins, \
            #    anchor_sample_masks, anchor_col_masks, anchor_opval_masks, anchor_join_masks, anchor_labels, anchor_template_labels, \
            #    positive_samples, positive_cols, positive_opvals, positive_joins, \
            #    positive_sample_masks, positive_col_masks, positive_opval_masks, positive_join_masks, positive_labels, positive_template_labels, \
            #    negative_samples, negative_cols, negative_opvals, negative_joins, \
            #    negative_sample_masks, negative_col_masks, negative_opval_masks, negative_join_masks, negative_labels, negative_template_labels = data_batch
            anchor_samples, anchor_predicates, anchor_joins, \
                anchor_sample_masks, anchor_predicate_masks, anchor_join_masks, anchor_labels, anchor_template_labels, \
                positive_samples, positive_predicates, positive_joins, \
                positive_sample_masks, positive_predicate_masks, positive_join_masks, positive_labels, positive_template_labels, \
                negative_samples, negative_predicates, negative_joins, \
                negative_sample_masks, negative_predicate_masks, negative_join_masks, negative_labels, negative_template_labels = data_batch
            if cuda == 'cuda':
                #anchor_samples, anchor_cols, anchor_opvals, anchor_joins = anchor_samples.cuda(), anchor_cols.cuda(), anchor_opvals.cuda(), anchor_joins.cuda()
                #anchor_sample_masks, anchor_col_masks, anchor_opval_masks, anchor_join_masks = anchor_sample_masks.cuda(), anchor_col_masks.cuda(), anchor_opval_masks.cuda(), anchor_join_masks.cuda()
                #anchor_labels, anchor_template_labels = anchor_labels.cuda(), anchor_template_labels.cuda()
                #positive_samples, positive_cols, positive_opvals, positive_joins = positive_samples.cuda(), positive_cols.cuda(), positive_opvals.cuda(), positive_joins.cuda()
                #positive_sample_masks, positive_col_masks, positive_opval_masks, positive_join_masks = positive_sample_masks.cuda(), positive_col_masks.cuda(), positive_opval_masks.cuda(), positive_join_masks.cuda()
                #positive_labels, positive_template_labels = positive_labels.cuda(), positive_template_labels.cuda()
                #negative_samples, negative_cols, negative_opvals, negative_joins = negative_samples.cuda(), negative_cols.cuda(), negative_opvals.cuda(), negative_joins.cuda()
                #negative_sample_masks, negative_col_masks, negative_opval_masks, negative_join_masks = negative_sample_masks.cuda(), negative_col_masks.cuda(), negative_opval_masks.cuda(), negative_join_masks.cuda()
                #negative_labels, negative_template_labels = negative_labels.cuda(), negative_template_labels.cuda()
                anchor_samples, anchor_predicates, anchor_joins = anchor_samples.cuda(), anchor_predicates.cuda(), anchor_joins.cuda()
                anchor_sample_masks, anchor_predicate_masks, anchor_join_masks = anchor_sample_masks.cuda(), anchor_predicate_masks.cuda(), anchor_join_masks.cuda()
                anchor_labels, anchor_template_labels = anchor_labels.cuda(), anchor_template_labels.cuda()
                positive_samples, positive_predicates, positive_joins = positive_samples.cuda(), positive_predicates.cuda(), positive_joins.cuda()
                positive_sample_masks, positive_predicate_masks, positive_join_masks = positive_sample_masks.cuda(), positive_predicate_masks.cuda(), positive_join_masks.cuda()
                positive_labels, positive_template_labels = positive_labels.cuda(), positive_template_labels.cuda()
                negative_samples, negative_predicates, negative_joins = negative_samples.cuda(), negative_predicates.cuda(), negative_joins.cuda()
                negative_sample_masks, negative_predicate_masks, negative_join_masks = negative_sample_masks.cuda(), negative_predicate_masks.cuda(), negative_join_masks.cuda()
                negative_labels, negative_template_labels = negative_labels.cuda(), negative_template_labels.cuda()

            optimizer.zero_grad()
            #anchor_pred, anchor_feature = model(anchor_samples, anchor_cols, anchor_opvals, anchor_joins, anchor_sample_masks, anchor_col_masks, anchor_opval_masks, anchor_join_masks)
            #positive_pred, positive_feature = model(positive_samples, positive_cols, positive_opvals, positive_joins, positive_sample_masks, positive_col_masks, positive_opval_masks, positive_join_masks)
            #negative_pred, negative_feature = model(negative_samples, negative_cols, negative_opvals, negative_joins, negative_sample_masks, negative_col_masks, negative_opval_masks, negative_join_masks)
            '''Inital forward pass to get the feature and labels, and compute the inital weight for each data'''
            if epoch % 10 == 10:
                with higher.innerloop_ctx(model, optimizer) as (meta_model, meta_optimizer):
                    #1. Update meta model on training data
                    anchor_pred, anchor_feature = meta_model(anchor_samples, anchor_predicates, anchor_joins, anchor_sample_masks, anchor_predicate_masks, anchor_join_masks)
                    #positive_pred, positive_feature = meta_model(positive_samples, positive_predicates, positive_joins, positive_sample_masks, positive_predicate_masks, positive_join_masks)
                    #negative_pred, negative_feature = meta_model(negative_samples, negative_predicates, negative_joins, negative_sample_masks, negative_predicate_masks, negative_join_masks)
                    qerror = qerror_loss(anchor_pred, anchor_labels.float(), min_val, max_val)
                    #csa = csa_loss(source_feature, target_feature, (source_template_labels == target_template_labels).float())
                    #triplet_loss = torch.nn.functional.triplet_margin_loss(anchor_feature, positive_feature, negative_feature, margin=0)
                    #loss = (1 - alpha) * qerror + alpha * csa
                    eps = torch.zeros(qerror.size(), requires_grad=True, device=cuda)
                    meta_loss = torch.sum(qerror * eps)
                    meta_optimizer.step(meta_loss)
                    #loss = (1 - alpha) * torch.mean(qerror) + alpha * triplet_loss 
                    #print(qerror)
                    #loss = qerror
                    #meta_model.zero_grad()
                    #2. Compute grads of eps on meta validation data

                    #print(l_f_meta)
                    #grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
                    #print(grads)
                    #meta_model.update_params(1e-3, source_params=grads)

                    #2nd forward pass and getting the gradients with respect to weight
                    l_g_meta = []
                    for test_data_batch in test_data_loader:
                        valid_samples, valid_predicates, valid_joins, \
                        valid_sample_masks, valid_predicate_masks, valid_join_masks, valid_labels, valid_template_labels, \
                        _, _, _, \
                        _, _, _, _, _, \
                        _, _, _, \
                        _, _, _, _, _ = test_data_batch
                        if cuda == 'cuda':
                            valid_samples, valid_predicates, valid_joins = valid_samples.cuda(), valid_predicates.cuda(), valid_joins.cuda()
                            valid_sample_masks, valid_predicate_masks, valid_join_masks = valid_sample_masks.cuda(), valid_predicate_masks.cuda(), valid_join_masks.cuda()
                            valid_labels, valid_template_labels = valid_labels.cuda(), valid_template_labels.cuda()
                        valid_pred, _ = meta_model(valid_samples, valid_predicates, valid_joins, valid_sample_masks, valid_predicate_masks, valid_join_masks)
                        l_g_meta.append(qerror_loss(valid_pred, valid_labels.float(), min_val, max_val))
                    l_g_meta = torch.mean(torch.cat(l_g_meta))
                    grad_eps = torch.autograd.grad(l_g_meta, eps)[0].detach()

                #compute and normalize the weight
                w_tilde = torch.clamp(-grad_eps,min=0)
                norm_c = torch.sum(w_tilde)
                if norm_c != 0:
                    w = w_tilde / norm_c
                else:
                    w = w_tilde
            else:
                w = 1 / anchor_samples.shape[0]
            '''compute the loss with the weight and update the parameters'''
            anchor_pred, anchor_feature = model(anchor_samples, anchor_predicates, anchor_joins, anchor_sample_masks, anchor_predicate_masks, anchor_join_masks)
            positive_pred, positive_feature = model(positive_samples, positive_predicates, positive_joins, positive_sample_masks, positive_predicate_masks, positive_join_masks)
            negative_pred, negative_feature = model(negative_samples, negative_predicates, negative_joins, negative_sample_masks, negative_predicate_masks, negative_join_masks)
            anchor_qerror = qerror_loss(anchor_pred, anchor_labels.float(), min_val, max_val)
            #print(anchor_qerror)
            positive_qerror = qerror_loss(positive_pred, positive_labels.float(), min_val, max_val)
            negative_qerror = qerror_loss(negative_pred, negative_labels.float(), min_val, max_val)
            #csa = csa_loss(source_feature, target_feature, (source_template_labels == target_template_labels).float())
            #triplet_loss = torch.nn.functional.triplet_margin_loss(anchor_feature, positive_feature, negative_feature, margin=0, reduction='none')
            cs_loss = ranking_loss(anchor_feature, anchor_labels.float(), anchor_pred, min_val, max_val)
            #loss = (1 - alpha) * torch.sum(w * (anchor_qerror + positive_qerror + negative_qerror)) + alpha * torch.sum(w * cs_loss)
            loss = w * torch.sum((anchor_qerror + positive_qerror + negative_qerror))
            #if contrastive is True:
            #    loss = torch.sum(w * anchor_qerror) + torch.sum(w * cs_loss)
            #else:
            #    loss = torch.sum(w * anchor_qerror)
            #tmpfp.write("{}\n".format(w.detach()))


            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        #if epoch % 10 == 0:
        #    model_file = save_folder + '/{}_adaptation_model_{}.pkl'.format(dataset, epoch)
        #    torch.save(model.state_dict(), model_file)
        #    print("\n Model save path: {}".format(model_file))
        preds_test, t_total = predict_triplet_query(model, test_data_loader, cuda, adaptation=True)
        preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val, cuda)
        labels_test_unnorm = unnormalize_labels(anchor_test_labels, min_val, max_val, cuda='cpu')
        #print("\nQ-Error validation set:")
        #median_error = print_qerror(preds_test_unnorm, labels_test_unnorm)
        #tmpfp.write(str(median_error) + '\n')
        print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))
    tmpfp.close()
    # Get final training and validation set predictions
    preds_train, t_total = predict_triplet_query(model, train_data_loader, cuda, adaptation=True)
    print("Prediction time per training sample: {}".format(t_total / len(anchor_train_labels) * 1000))

    preds_test, t_total = predict_triplet_query(model, test_data_loader, cuda, adaptation=True)
    print("Prediction time per validation sample: {}".format(t_total / len(anchor_test_labels) * 1000))

    # Unnormalize
    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val, cuda)
    labels_train_unnorm = unnormalize_labels(anchor_train_labels, min_val, max_val, cuda='cpu')

    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val, cuda)
    labels_test_unnorm = unnormalize_labels(anchor_test_labels, min_val, max_val, cuda='cpu')

    # Print metrics
    print("\nQ-Error training set:")
    print_qerror(preds_train_unnorm, labels_train_unnorm)

    print("\nQ-Error validation set:")
    print_qerror(preds_test_unnorm, labels_test_unnorm)
    print("")
    if directly_adapt:
        model_file = save_folder + '/{}_directly_adaptation_metamodel_{}_{}.pkl'.format(dataset, alpha, query_num)
    else:
        model_file = save_folder + '/{}_adaptation_model_{}_{}'.format(dataset, alpha, query_num)
        if contrastive is True:
            model_file += "_contrastive"
        if meta is True:
            model_file += "_meta"
        model_file += '.pkl'
    torch.save(model.state_dict(), model_file)
    print("\n Model save path: ")
    print(model_file)

    structure_file = save_folder + '/{}_structure_adaptation'.format(dataset)
    if contrastive is True:
        structure_file += "_contrastive"
    if meta is True:
        structure_file += "_meta"
    structure_file += ".pkl"
    with open(structure_file, 'wb') as fp:
        save_dicts = {'column_min_max_vals': column_min_max_vals, 
                      'table2vec': table2vec,
                      'join2vec': join2vec,
                      'column2vec': column2vec,
                      'op2vec': op2vec,
                      'vec2table': vec2table,
                      'vec2join': vec2join,
                      'vec2column': vec2column,
                      'vec2op': vec2op, 
                      'vec2table': vec2table,
                      'vec2join': vec2join,
                      'vec2column': vec2column,
                      'vec2op': vec2op, 
                      #'max_cols': max_num_cols,
                      #'max_opvals': max_num_opvals,
                      'max_predicates': max_num_predicates,
                      'max_joins': max_num_joins,
                      'min_val': min_val,
                      'max_val': max_val,
                      'templates': templates,
                      'template_distribution': template_distribution}
        pickle.dump(save_dicts, fp)
    return


def train_triplet(dataset, query_file, min_max_file, save_folder, bitmap_file, adaptation_query_file, adaptation_bitmap_file, conn, cursor, num_samples=0, cuda='cuda', hid_units=256, batch_size=1024, num_epochs=10, alpha=0.25, lr=0.25, repeat=10, contrastive=True, meta=False, adaptation_indices=None, template_contrastive=False, static_target_weight=None, initial_model_file=None, robust=False, target_only=False, temperature=2.0, random_contrastive=False, evaluate_after_training=True):
    #dicts, idxdicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_cols, max_num_opvals, train_data, \
    #     test_data, templates, template_distribution, template_labels_train, template_labels_test = get_triplet_train_datasets(query_file, bitmap_file, min_max_file, num_samples)
    dicts, idxdicts, column_min_max_vals, min_val, max_val, labels_train, labels_test, max_num_joins, max_num_predicates, train_data, \
         test_data, templates, template_distribution, template_labels_train, template_labels_test = get_triplet_train_datasets(
             query_file, bitmap_file, adaptation_query_file,
             adaptation_bitmap_file, min_max_file, num_samples, repeat,
             adaptation_indices=adaptation_indices)
    if target_only:
        # Target trains on the selected target feedback only. Reuse that set
        # as the non-meta validation tensor because the normal Adapt training
        # loop expects both tensor collections; it is never used for model
        # selection or final evaluation.
        train_data = test_data
        labels_train = labels_test
        template_labels_train = template_labels_test
    
    table2vec, column2vec, op2vec, join2vec = dicts
    vec2table, vec2column, vec2op, vec2join = idxdicts
    #statistics_max = [max_num_cols, max_num_opvals, max_num_joins]
    statistics_max = [max_num_predicates, max_num_joins]

    # Train model
    sample_feats = len(table2vec) + num_samples
    col_feats = len(column2vec)
    opval_feats = len(op2vec) + 1
    join_feats = len(join2vec)

    model = SetConv(sample_feats, col_feats, opval_feats, join_feats,
                    hid_units, robust=robust)
    if initial_model_file is not None:
        if not os.path.exists(initial_model_file):
            raise FileNotFoundError(
                "Sequential warm-start checkpoint not found: {}".format(
                    initial_model_file))
        model.load_state_dict(torch.load(
            initial_model_file, map_location=torch.device(cuda)))
        print("Warm-start model pkl from {}".format(initial_model_file))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if cuda == 'cuda':
        model.cuda()

    model.train()
    anchor_labels_train = labels_train
    anchor_labels_test = labels_test
    train_dataset, test_dataset = generate_epoch_data(train_data, test_data, labels_train, labels_test, template_labels_train, template_labels_test)
    prediction_train_dataset = train_dataset
    if static_target_weight is not None:
        target_count = len(labels_test)
        domain_weights = torch.ones(len(train_dataset), dtype=torch.float32)
        if target_count:
            domain_weights[-target_count:] = float(static_target_weight)
        train_dataset = torch.utils.data.TensorDataset(
            *train_dataset.tensors, domain_weights)
    print("Contrastive: ", contrastive)
    print("Meta: ", meta)
    # A DataLoader reshuffles on every iteration, so it need not be rebuilt
    # for every epoch.
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=(cuda == 'cuda'))
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=(cuda == 'cuda'))
    if cuda == 'cuda':
        torch.cuda.synchronize()
    optimization_started = time.perf_counter()
    for epoch in range(num_epochs):
        loss_total = 0.
        for batch_idx, data_batch in enumerate(train_data_loader):
            if static_target_weight is not None:
                data_batch, batch_static_weights = data_batch[:-1], data_batch[-1]
            anchor_samples, anchor_predicates, anchor_joins, anchor_sample_masks, \
            anchor_predicate_masks, anchor_join_masks, anchor_labels, anchor_template_labels, \
            positive_samples, positive_predicates, positive_joins, positive_sample_masks, \
            positive_predicate_masks, positive_join_masks, positive_labels, positive_template_labels, \
            negative_samples, negative_predicates, negative_joins, negative_sample_masks, \
            negative_predicate_masks, negative_join_masks, negative_labels, negative_template_labels = data_batch
         
            if cuda == 'cuda':
                anchor_samples, anchor_predicates, anchor_joins = anchor_samples.cuda(non_blocking=True), anchor_predicates.cuda(non_blocking=True), anchor_joins.cuda(non_blocking=True)
                anchor_sample_masks, anchor_predicate_masks, anchor_join_masks = anchor_sample_masks.cuda(non_blocking=True), anchor_predicate_masks.cuda(non_blocking=True), anchor_join_masks.cuda(non_blocking=True)
                anchor_labels = anchor_labels.cuda(non_blocking=True)
                anchor_template_labels = anchor_template_labels.cuda(non_blocking=True)
                # These branches are legacy triplet inputs. Their outputs do
                # not enter the current objective. Keep the calls only for
                # Robust-MSCN, where they consume dropout RNG and therefore
                # affect exact run reproducibility.
                if robust:
                    positive_samples, positive_predicates, positive_joins = positive_samples.cuda(non_blocking=True), positive_predicates.cuda(non_blocking=True), positive_joins.cuda(non_blocking=True)
                    positive_sample_masks, positive_predicate_masks, positive_join_masks = positive_sample_masks.cuda(non_blocking=True), positive_predicate_masks.cuda(non_blocking=True), positive_join_masks.cuda(non_blocking=True)
                    negative_samples, negative_predicates, negative_joins = negative_samples.cuda(non_blocking=True), negative_predicates.cuda(non_blocking=True), negative_joins.cuda(non_blocking=True)
                    negative_sample_masks, negative_predicate_masks, negative_join_masks = negative_sample_masks.cuda(non_blocking=True), negative_predicate_masks.cuda(non_blocking=True), negative_join_masks.cuda(non_blocking=True)
            optimizer.zero_grad()
            if static_target_weight is not None:
                if cuda == 'cuda':
                    batch_static_weights = batch_static_weights.cuda()
                w = batch_static_weights / batch_static_weights.sum()
            else:
                w = 1 / anchor_samples.shape[0]
            if meta is True:
                interval = 0
            else:
                interval = 10 
            if epoch % 10 == interval:
                with higher.innerloop_ctx(model, optimizer) as (meta_model, meta_optimizer):
                #1. Update meta model on training data
                    anchor_pred, anchor_feature = meta_model(anchor_samples, anchor_predicates, anchor_joins, anchor_sample_masks, anchor_predicate_masks, anchor_join_masks)
                    qerror = qerror_loss(anchor_pred, anchor_labels.float(), min_val, max_val)
                    #csa = csa_loss(source_feature, target_feature, (source_template_labels == target_template_labels).float())
                    #triplet_loss = torch.nn.functional.triplet_margin_loss(anchor_feature, positive_feature, negative_feature, margin=0)
                    #loss = (1 - alpha) * qerror + alpha * csa
                    eps = torch.zeros(qerror.size(), requires_grad=True, device=cuda)
                    meta_loss = torch.sum(qerror * eps)
                    meta_optimizer.step(meta_loss)
                    
                    #2nd forward pass and getting the gradients with respect to weight
                    l_g_meta = []
                    for test_data_batch in test_data_loader:
                        valid_samples, valid_predicates, valid_joins, \
                        valid_sample_masks, valid_predicate_masks, valid_join_masks, valid_labels, valid_template_labels, \
                        _, _, _, \
                        _, _, _, _, _, \
                        _, _, _, \
                        _, _, _, _, _ = test_data_batch
                        if cuda == 'cuda':
                            valid_samples, valid_predicates, valid_joins = valid_samples.cuda(), valid_predicates.cuda(), valid_joins.cuda()
                            valid_sample_masks, valid_predicate_masks, valid_join_masks = valid_sample_masks.cuda(), valid_predicate_masks.cuda(), valid_join_masks.cuda()
                            valid_labels, valid_template_labels = valid_labels.cuda(), valid_template_labels.cuda()
                        valid_pred, _ = meta_model(valid_samples, valid_predicates, valid_joins, valid_sample_masks, valid_predicate_masks, valid_join_masks)
                        l_g_meta.append(qerror_loss(valid_pred, valid_labels.float(), min_val, max_val))
                    l_g_meta = torch.mean(torch.cat(l_g_meta))
                    grad_eps = torch.autograd.grad(l_g_meta, eps)[0].detach()

                #compute and normalize the weight
                w_tilde = torch.clamp(-grad_eps,min=0)
                norm_c = torch.sum(w_tilde)
                if norm_c != 0:
                    w = w_tilde / norm_c
                else:
                    w = w_tilde
            #else:
            #    w = 1 / anchor_samples.shape[0]
            #anchor_pred, anchor_feature = model(anchor_samples, anchor_cols, anchor_opvals, anchor_joins, anchor_sample_masks, anchor_col_masks, anchor_opval_masks, anchor_join_masks)
            #positive_pred, positive_feature = model(positive_samples, positive_cols, positive_opvals, positive_joins, positive_sample_masks, positive_col_masks, positive_opval_masks, positive_join_masks)
            #negative_pred, negative_feature = model(negative_samples, negative_cols, negative_opvals, negative_joins, negative_sample_masks, negative_col_masks, negative_opval_masks, negative_join_masks)
            anchor_pred, anchor_feature = model(anchor_samples, anchor_predicates, anchor_joins, anchor_sample_masks, anchor_predicate_masks, anchor_join_masks)
            if robust:
                # Preserve the original dropout RNG stream for robust runs.
                model(positive_samples, positive_predicates, positive_joins, positive_sample_masks, positive_predicate_masks, positive_join_masks)
                model(negative_samples, negative_predicates, negative_joins, negative_sample_masks, negative_predicate_masks, negative_join_masks)
            anchor_qerror = qerror_loss(anchor_pred, anchor_labels.float(), min_val, max_val)
            #triplet_loss = torch.nn.functional.triplet_margin_loss(anchor_feature, positive_feature, negative_feature, margin=1)
            if template_contrastive:
                cs_loss = template_contrastive_loss(
                    anchor_feature, anchor_template_labels, temperature=temperature)
            elif contrastive:
                contrastive_labels = anchor_labels.float()
                if random_contrastive:
                    contrastive_labels = contrastive_labels[
                        torch.randperm(contrastive_labels.shape[0], device=contrastive_labels.device)]
                cs_loss = ranking_loss(anchor_feature, contrastive_labels, anchor_pred, min_val, max_val, temperature=temperature)
            #loss = (1 - alpha) * (target_qerror) + alpha * csa
            #loss = (1 - alpha) * torch.mean(anchor_qerror) + alpha * triplet_loss
            if contrastive is True or template_contrastive:
                loss = alpha * torch.sum(w * cs_loss) + (1 - alpha) * torch.sum(w * anchor_qerror)
            else:
                loss = torch.sum(w * anchor_qerror)
            #loss = alpha * triplet_loss
            #loss = torch.mean(anchor_qerror)
            #loss = (1 - alpha) * torch.sum(w * anchor_qerror) + alpha * torch.sum(w * cs_loss)
            #loss = target_qerror
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        print("Epoch {}, loss: {}".format(epoch, loss_total / len(train_data_loader)))
    if cuda == 'cuda':
        torch.cuda.synchronize()
    print("Optimization training time: {:.6f}s".format(
        time.perf_counter() - optimization_started))
    # Get final training and validation set predictions
    if evaluate_after_training:
        train_data_loader = DataLoader(prediction_train_dataset, batch_size=batch_size)
        preds_train, t_total = predict_triplet_query(model, train_data_loader, cuda, adaptation=True)
        print("Prediction time per training sample: {}".format(t_total / len(anchor_labels_train) * 1000))

        preds_test, t_total = predict_triplet_query(model, test_data_loader, cuda, adaptation=True)
        print("Prediction time per validation sample: {}".format(t_total / len(anchor_labels_test) * 1000))

        # Unnormalize
        preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val, cuda)
        labels_train_unnorm = unnormalize_labels(anchor_labels_train, min_val, max_val, cuda='cpu')

        preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val, cuda)
        labels_test_unnorm = unnormalize_labels(anchor_labels_test, min_val, max_val, cuda='cpu')

        # Print metrics
        print("\nQ-Error training set:")
        print_qerror(preds_train_unnorm, labels_train_unnorm)

        print("\nQ-Error validation set:")
        print_qerror(preds_test_unnorm, labels_test_unnorm)
        print("")

    model_file = save_folder + '/{}_model_{}'.format(dataset, alpha)
    if contrastive is True:
        model_file += "_contrastive"
    if meta is True:
        model_file += "_meta"
    model_file += ".pkl"
    torch.save(model.state_dict(), model_file)
    print("\n Model save path: ")
    print(model_file)

    structure_file = save_folder + '/{}_structure'.format(dataset)
    if contrastive is True:
        structure_file += "_contrastive"
    if meta is True:
        structure_file += "_meta"
    structure_file += ".pkl"
    with open(structure_file, 'wb') as fp:
        save_dicts = {'column_min_max_vals': column_min_max_vals, 
                      'table2vec': table2vec,
                      'join2vec': join2vec,
                      'column2vec': column2vec,
                      'op2vec': op2vec,
                      'vec2table': vec2table,
                      'vec2join': vec2join,
                      'vec2column': vec2column,
                      'vec2op': vec2op, 
                      #'max_cols': max_num_cols,
                      #'max_opvals': max_num_opvals,
                      'max_predicates': max_num_predicates,
                      'max_joins': max_num_joins,
                      'min_val': min_val,
                      'max_val': max_val,
                      'templates': templates,
                      'template_distribution': template_distribution}
        pickle.dump(save_dicts, fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mode", type=str, default='train', help='train, predict or adaptation')
    parser.add_argument("--dataset", type=str, default='imdb', help='dataset')
    parser.add_argument("--device", type=str, default='cpu', help='cpu or cuda')
    parser.add_argument("--batch_size", type=int, default=512, help='batch size')
    parser.add_argument("--epochs", type=int, default=20, help='number of epochs')
    parser.add_argument("--num_samples", type=int, default=1000, help='number of samples in bitmaps')
    parser.add_argument("--alpha", type=float, default=0.5, help='hyperparameter')
    parser.add_argument("--temperature", type=float, default=2.0,
                        help='contrastive softmax temperature')
    parser.add_argument("--query_num", type=int, default=50000, help='number of queries in adaptation')
    parser.add_argument("--lr", type=float, default=0.001, help='hyperparameter')
    parser.add_argument("--repeat", type=int, default=10, help='hyperparameter')
    parser.add_argument("--adaptation_predict", action='store_true', help='prediction with adaptation')
    parser.add_argument("--directly_adapt", action='store_true', help='use adaptation directly')
    parser.add_argument("--contrastive", action='store_true', help='use contrastive learning')
    parser.add_argument("--template_contrastive", action='store_true',
                        help='use query-template supervised contrastive learning')
    parser.add_argument("--random_contrastive", action='store_true',
                        help='shuffle output relationships as a random-pair control')
    parser.add_argument("--static_target_weight", type=float,
                        help="fixed per-target weight; enables Static-Upweight")
    parser.add_argument("--static_train_indices",
                        help="comma-separated adaptation IDs for an inner-CV fold")
    parser.add_argument("--static_validation_indices",
                        help="comma-separated held-out adaptation IDs for inner CV")
    parser.add_argument("--meta", action="store_true", help='use meta learning')
    parser.add_argument("--robust", action="store_true",
                        help="enable Robust-MSCN feature masking/dropout")
    parser.add_argument("--target_only", action="store_true",
                        help="train using only the selected target queries")
    parser.add_argument("--use_all_candidates", action="store_true",
                        help="use the complete fixed adaptation candidate pool")
    parser.add_argument("--seed", type=int, default=None,
                        help="global seed; enables deterministic target sampling when set")
    parser.add_argument("--adaptation_budget", type=int, default=10,
                        help="number of target queries sampled in seeded mode")
    parser.add_argument("--output_root", type=str, default="results",
                        help="root directory for isolated seeded artifacts")
    parser.add_argument("--method_name", type=str, default=None,
                        help="method label used to isolate and audit artifacts")
    parser.add_argument("--prepare_selection_only", action="store_true",
                        help="write and validate seeded manifests without training")
    parser.add_argument("--target_candidate_fraction", type=float, default=0.2,
                        help="fixed fraction reserved for adaptation candidates")
    parser.add_argument("--split_seed", type=int, default=20260923,
                        help="seed for the fixed candidate/evaluation split")
    parser.add_argument("--selection_indices", type=str,
                        help="explicit comma-separated target IDs selected by an external policy")
    parser.add_argument("--train_only", action="store_true",
                        help="train and save the model without running prediction")
    args = parser.parse_args()
    if args.seed is not None:
        set_global_seed(args.seed)
    print('dataset: ', args.dataset)
    params = {
        'lr': args.lr,
        'alpha': args.alpha,
        'repeat': args.repeat,
    }
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)
    if args.dataset == 'imdb':
        train_query_file, min_max_file, predict_query_file, result_file, save_folder, adaptation_query_file, validation_file, train_bitmap_file, predicate_bitmap_file, adaptation_bitmap_file, tables = load_imdb_dataset()
    elif args.dataset == 'higss':
        train_query_file, min_max_file, predict_query_file, result_file, save_folder, adaptation_query_file, validation_file, train_bitmap_file, predicate_bitmap_file, adaptation_bitmap_file, tables = load_higss_dataset()
    elif args.dataset == 'job-light-ranges':
        train_query_file, min_max_file, predict_query_file, result_file, save_folder, adaptation_query_file, validation_file, train_bitmap_file, predicate_bitmap_file, adaptation_bitmap_file, tables = load_job_dataset()
    elif args.dataset == 'higgs_full':
        train_query_file, min_max_file, predict_query_file, result_file, save_folder, adaptation_query_file, validation_file, train_bitmap_file, predicate_bitmap_file, adaptation_bitmap_file, tables = load_higgs_full_dataset()
    elif args.dataset == 'forest':
        train_query_file, min_max_file, predict_query_file, result_file, save_folder, adaptation_query_file, validation_file, train_bitmap_file, predicate_bitmap_file, adaptation_bitmap_file, tables = load_forest_dataset()
    elif args.dataset == 'stats':
        train_query_file, min_max_file, predict_query_file, result_file, save_folder, adaptation_query_file, validation_file, train_bitmap_file, predicate_bitmap_file, adaptation_bitmap_file, tables = load_stats_dataset()
    elif args.dataset == 'job':
        train_query_file, min_max_file, predict_query_file, result_file, save_folder, adaptation_query_file, validation_file, train_bitmap_file, predicate_bitmap_file, adaptation_bitmap_file, tables = load_job_ceb_dataset()
    elif args.dataset == 'dsb':
        train_query_file, min_max_file, predict_query_file, result_file, save_folder, adaptation_query_file, validation_file, train_bitmap_file, predicate_bitmap_file, adaptation_bitmap_file, tables = load_dsb_dataset()
    #conn, cursor = pg_initialize(args.dataset, tables)
    conn = 0
    cursor = 0
    adaptation_indices = None
    static_validation_indices = None
    artifact_folder = save_folder
    if args.seed is not None:
        # Use the target prediction workload as the common candidate pool.
        # Selected adaptation queries are excluded from evaluation below.
        adaptation_query_file = predict_query_file
        adaptation_bitmap_file = predicate_bitmap_file
        split, split_file = load_or_create_target_split(
            args.output_root, args.dataset, adaptation_query_file,
            args.target_candidate_fraction, args.split_seed)
        selection, selection_file = load_or_create_selection_manifest(
            args.output_root, args.dataset, args.adaptation_budget,
            args.seed, adaptation_query_file, split, split_file)
        if args.selection_indices:
            explicit_indices = parse_index_list(args.selection_indices)
            if (explicit_indices is None or
                    len(explicit_indices) != args.adaptation_budget or
                    len(set(explicit_indices)) != len(explicit_indices)):
                raise ValueError("Explicit selection must contain exactly adaptation_budget unique IDs")
            if not set(explicit_indices).issubset(set(split["candidate_indices"])):
                raise ValueError("Explicit selection contains IDs outside the fixed candidate pool")
            selection = dict(selection)
            selection["adaptation_indices"] = sorted(explicit_indices)
            selection["adaptation_budget"] = len(explicit_indices)
            selection["selection_id"] = indices_hash(selection["adaptation_indices"])
            selection["selection_policy"] = "external"
            external_dir = os.path.join(args.output_root, args.dataset, "external_selections")
            os.makedirs(external_dir, exist_ok=True)
            selection_file = os.path.join(
                external_dir, "{}_seed_{:03d}.json".format(
                    args.method_name or "external", args.seed))
            with open(selection_file, "w") as selection_stream:
                json.dump(selection, selection_stream, indent=2, sort_keys=True)
        if args.use_all_candidates:
            selection = dict(selection)
            selection["adaptation_indices"] = split["candidate_indices"]
            selection["adaptation_budget"] = len(split["candidate_indices"])
            selection["selection_id"] = indices_hash(split["candidate_indices"])
        pool_size = selection["adaptation_pool_size"]
        adaptation_indices = selection["adaptation_indices"]
        static_train_indices = parse_index_list(args.static_train_indices)
        static_validation_indices = parse_index_list(args.static_validation_indices)
        if static_train_indices is not None:
            if args.static_target_weight is None or static_validation_indices is None:
                raise ValueError("Static inner CV requires weight, train IDs, and validation IDs")
            selected_set = set(adaptation_indices)
            if (set(static_train_indices) | set(static_validation_indices) != selected_set or
                    set(static_train_indices) & set(static_validation_indices)):
                raise ValueError("Static CV train/validation IDs must partition adaptation IDs")
            adaptation_indices = static_train_indices
        method_name = resolve_method_name(
            args.method_name, args.contrastive, args.meta)
        artifact_folder = os.path.join(
            args.output_root, args.dataset,
            "budget_{}".format(args.adaptation_budget),
            method_name,
            "seed_{:03d}".format(args.seed))
        os.makedirs(artifact_folder, exist_ok=True)
        result_file = os.path.join(artifact_folder, "predictions.csv")
        manifest = {
            "dataset": args.dataset,
            "seed": args.seed,
            "adaptation_budget": args.adaptation_budget,
            "adaptation_pool_file": os.path.abspath(adaptation_query_file),
            "adaptation_pool_size": pool_size,
            "adaptation_indices": adaptation_indices,
            "full_adaptation_indices": selection["adaptation_indices"],
            "static_validation_indices": static_validation_indices,
            "selection_id": selection["selection_id"],
            "selection_manifest": os.path.abspath(selection_file),
            "method": method_name,
            "evaluation_file": os.path.abspath(predict_query_file),
            "evaluation_is_fixed_across_seeds": True,
            "evaluation_size": len(split["evaluation_indices"]),
            "evaluation_id": split["evaluation_id"],
            "split_id": split["split_id"],
            "split_manifest": os.path.abspath(split_file),
            "training": {
                "alpha": params["alpha"],
                "batch_size": args.batch_size,
                "contrastive": args.contrastive,
                "template_contrastive": args.template_contrastive,
                "random_contrastive": args.random_contrastive,
                "temperature": args.temperature,
                "static_target_weight": args.static_target_weight,
                "device": args.device,
                "epochs": args.epochs,
                "learning_rate": params["lr"],
                "meta": args.meta,
                "robust": args.robust,
                "target_only": args.target_only,
                "use_all_candidates": args.use_all_candidates,
                "num_materialized_samples": args.num_samples,
                "target_repeat": params["repeat"],
            },
        }
        with open(os.path.join(artifact_folder, "manifest.json"), "w") as manifest_stream:
            json.dump(manifest, manifest_stream, indent=2, sort_keys=True)
        print("Seeded adaptation manifest:", manifest)
        if args.prepare_selection_only:
            return
    if args.train_mode == 'train':
        #train(args.dataset, train_query_file, min_max_file, save_folder, train_bitmap_file, num_samples=args.num_samples, num_epochs=args.epochs, cuda=args.device, batch_size=args.batch_size, alpha=args.alpha)
        start_time = time.time()
        train_triplet(args.dataset, train_query_file, min_max_file, artifact_folder, train_bitmap_file, adaptation_query_file, adaptation_bitmap_file, conn, cursor, num_samples=args.num_samples, cuda=args.device, batch_size=args.batch_size, num_epochs=args.epochs, alpha=params['alpha'], contrastive=args.contrastive, meta=args.meta, lr=params['lr'], repeat=params['repeat'], adaptation_indices=adaptation_indices, template_contrastive=args.template_contrastive, static_target_weight=args.static_target_weight, robust=args.robust, target_only=args.target_only, temperature=args.temperature, random_contrastive=args.random_contrastive, evaluate_after_training=not args.train_only)
        end_time = time.time()
        print("Total training time: {}s".format(end_time - start_time))
        #if args.contrastive is True:
        #    adaptation_triplet(args.dataset, train_query_file, adaptation_query_file, save_folder, train_bitmap_file, adaptation_bitmap_file, conn, cursor, cuda=args.device, batch_size=args.batch_size, num_epochs=args.epochs, num_samples=args.num_samples, alpha=params['alpha'], query_num=args.query_num, directly_adapt=args.directly_adapt, contrastive=args.contrastive, meta=args.meta)
        if args.train_only:
            return
        if args.seed is not None and static_validation_indices is not None:
            evaluation_indices = static_validation_indices
        else:
            evaluation_indices = split["evaluation_indices"] if args.seed is not None else None
        predict(args.dataset, predict_query_file, result_file, artifact_folder, predicate_bitmap_file, num_samples=args.num_samples, cuda=args.device, adaptation=args.adaptation_predict, batch_size=args.batch_size, alpha=params['alpha'], query_num=args.query_num, directly_adapt=args.directly_adapt, contrastive=args.contrastive, meta=args.meta, include_indices=evaluation_indices)
    elif args.train_mode == 'predict':
        predict(args.dataset, predict_query_file, result_file, save_folder, predicate_bitmap_file, num_samples=args.num_samples, cuda=args.device, adaptation=args.adaptation_predict, batch_size=args.batch_size, alpha=params['alpha'], query_num=args.query_num, directly_adapt=args.directly_adapt, contrastive=args.contrastive, meta=args.meta)
    elif args.train_mode == 'adaptation':
        adaptation_triplet(args.dataset, train_query_file, adaptation_query_file, save_folder, train_bitmap_file, adaptation_bitmap_file, conn, cursor, cuda=args.device, batch_size=args.batch_size, num_epochs=args.epochs, num_samples=args.num_samples, alpha=params['alpha'], query_num=args.query_num, directly_adapt=args.directly_adapt, contrastive=args.contrastive, meta=args.meta)

if __name__ == '__main__':
    main()


