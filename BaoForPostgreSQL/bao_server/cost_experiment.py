"""Shared selection manifests and reproducibility helpers for Bao cost runs."""

import hashlib
import json
import os
import random

import numpy as np
import torch

from train import load_aligned_experience


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        pass


def ids_hash(ids):
    value = ",".join(str(item) for item in ids).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def file_fingerprint(plan_dir, label_file, ids):
    digest = hashlib.sha256()
    labels = []
    with open(label_file) as stream:
        labels = [line.strip() for line in stream]
    for plan_id in ids:
        path = os.path.join(plan_dir, "sql{}_plan.json".format(plan_id))
        digest.update(str(plan_id).encode("utf-8"))
        with open(path, "rb") as stream:
            digest.update(stream.read())
        digest.update(labels[plan_id].encode("utf-8"))
    return digest.hexdigest()


def load_or_create_selection(output_root, dataset, budget, seed,
                             plan_dir, label_file):
    _, _, pool_ids, alignment = load_aligned_experience(plan_dir, label_file)
    if budget <= 0 or budget > len(pool_ids):
        raise ValueError("Invalid adaptation budget {} for pool size {}".format(
            budget, len(pool_ids)))
    rng = np.random.default_rng(seed)
    selected = sorted(int(x) for x in rng.choice(
        pool_ids, size=budget, replace=False))
    selection_dir = os.path.join(
        output_root, dataset, "budget_{}".format(budget), "selections")
    os.makedirs(selection_dir, exist_ok=True)
    path = os.path.join(selection_dir, "seed_{:03d}.json".format(seed))
    expected = {
        "dataset": dataset,
        "seed": seed,
        "adaptation_budget": budget,
        "adaptation_pool_file": os.path.abspath(plan_dir),
        "adaptation_label_file": os.path.abspath(label_file),
        "adaptation_pool_size": len(pool_ids),
        "adaptation_indices": selected,
        "selection_id": ids_hash(selected),
        "alignment": alignment,
    }
    if os.path.exists(path):
        with open(path) as stream:
            existing = json.load(stream)
        if existing != expected:
            raise RuntimeError("Existing selection manifest differs: {}".format(path))
        return existing, path
    with open(path, "w") as stream:
        json.dump(expected, stream, indent=2, sort_keys=True)
    return expected, path


def evaluation_identity(plan_dir, label_file):
    _, _, ids, alignment = load_aligned_experience(plan_dir, label_file)
    return file_fingerprint(plan_dir, label_file, ids), alignment
