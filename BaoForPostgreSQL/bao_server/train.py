import storage
import model
import os
import shutil
import reg_blocker
import json
import re
try:
    import nni
except Exception:
    class _NniFallback:
        @staticmethod
        def report_final_result(_result):
            return None
    nni = _NniFallback()

import numpy as np

class BaoTrainingException(Exception):
    pass

def train_and_swap(fn, old, tmp, verbose=False):
    if os.path.exists(fn):
        old_model = model.BaoRegression(have_cache_data=True)
        old_model.load(fn)
    else:
        old_model = None

    new_model = train_and_save_model(tmp, verbose=verbose)
    max_retries = 5
    current_retry = 1
    while not reg_blocker.should_replace_model(old_model, new_model):
        if current_retry >= max_retries == 0:
            print("Could not train model with better regression profile.")
            return
        
        print("New model rejected when compared with old model. "
              + "Trying to retrain with emphasis on regressions.")
        print("Retry #", current_retry)
        new_model = train_and_save_model(tmp, verbose=verbose,
                                         emphasize_experiments=current_retry)
        current_retry += 1

    if os.path.exists(fn):
        shutil.rmtree(old, ignore_errors=True)
        os.rename(fn, old)
    os.rename(tmp, fn)

def train_and_save_model(fn, verbose=True, emphasize_experiments=0, alpha=0.25):
    all_experience = storage.experience()

    for _ in range(emphasize_experiments):
        all_experience.extend(storage.experiment_experience())
    print(all_experience[0])    
    x = [i[0] for i in all_experience]
    y = [i[1] for i in all_experience]        
    
    if not all_experience:
        raise BaoTrainingException("Cannot train a Bao model with no experience")
    
    if len(all_experience) < 20:
        print("Warning: trying to train a Bao model with fewer than 20 datapoints.")

    reg = model.BaoRegression(have_cache_data=True, verbose=verbose, alpha=alpha)
    reg.fit(x, y)
    reg.save(fn)
    return reg

def load_experience(filename):
    all_experience = [] 
    filenames = os.listdir(filename)
    for i in range(len(filenames)):
        with open(filename + "/sql{}_plan.json".format(i), 'r') as fp:
            data = json.load(fp)
            all_experience.append(data)
    return all_experience

def load_labels(filename):
    with open(filename) as fp:
        lines = fp.readlines()
        all_labels = [float(i) for i in lines]
    return all_labels


def load_aligned_experience(plan_dir, label_file, selected_ids=None):
    """Load labels by explicit sql<ID> plan IDs and audit legacy extra labels."""
    pattern = re.compile(r"^sql(\d+)_plan\.json$")
    id_to_path = {}
    for name in os.listdir(plan_dir):
        match = pattern.match(name)
        if match:
            plan_id = int(match.group(1))
            if plan_id in id_to_path:
                raise ValueError("Duplicate plan ID {} in {}".format(plan_id, plan_dir))
            id_to_path[plan_id] = os.path.join(plan_dir, name)
    if not id_to_path:
        raise ValueError("No sql<ID>_plan.json files in {}".format(plan_dir))
    labels = load_labels(label_file)
    all_ids = sorted(id_to_path)
    missing_labels = [plan_id for plan_id in all_ids if plan_id >= len(labels)]
    if missing_labels:
        raise ValueError("Missing labels for plan IDs {} in {}".format(
            missing_labels[:10], label_file))
    max_id = max(all_ids)
    missing_plans = sorted(set(range(max_id + 1)) - set(all_ids))
    if missing_plans:
        raise ValueError("Non-contiguous plan IDs {} in {}".format(
            missing_plans[:10], plan_dir))
    ignored_label_ids = list(range(max_id + 1, len(labels)))
    ids = all_ids if selected_ids is None else [int(x) for x in selected_ids]
    unknown = sorted(set(ids) - set(all_ids))
    if unknown:
        raise ValueError("Selected plan IDs do not exist: {}".format(unknown[:10]))
    experiences = []
    for plan_id in ids:
        with open(id_to_path[plan_id]) as stream:
            experiences.append(json.load(stream))
    metadata = {
        "plan_dir": os.path.abspath(plan_dir),
        "label_file": os.path.abspath(label_file),
        "plan_count": len(all_ids),
        "label_count": len(labels),
        "selected_count": len(ids),
        "ignored_trailing_label_ids": ignored_label_ids,
    }
    return experiences, [labels[plan_id] for plan_id in ids], ids, metadata


def sql_template_signature(sql):
    """Remove constants while retaining tables, joins, columns and operators."""
    normalized = " ".join(sql.strip().lower().split()).rstrip(";")
    match = re.search(r"\bfrom\b(.*?)(?:\bwhere\b(.*))?$", normalized)
    if not match:
        raise ValueError("Cannot extract FROM clause from SQL: {}".format(sql))
    tables = sorted(part.strip() for part in match.group(1).split(","))
    joins, predicates = [], []
    where = match.group(2) or ""
    for condition in re.split(r"\s+and\s+", where):
        condition = condition.strip().strip("()")
        comparison = re.match(r"(.+?)\s*(<=|>=|<>|!=|=|<|>)\s*(.+)$", condition)
        if not comparison:
            if condition:
                predicates.append(re.sub(r"'(?:''|[^'])*'|[-+]?\d+(?:\.\d+)?", "?", condition))
            continue
        left, operator, right = (item.strip() for item in comparison.groups())
        is_column = re.fullmatch(r"[a-z_][\w$]*\.[a-z_][\w$]*", right) is not None
        if is_column:
            if operator == "=":
                endpoints = sorted((left, right))
                joins.append(endpoints[0] + "=" + endpoints[1])
            else:
                joins.append(left + operator + right)
        else:
            predicates.append(left + operator)
    return "tables:" + ",".join(tables) + "#joins:" + ",".join(sorted(joins)) + \
        "#predicates:" + ",".join(sorted(predicates))


def load_sql_template_signatures(sql_file, ids):
    with open(sql_file) as stream:
        statements = [line.strip() for line in stream if line.strip()]
    missing = [plan_id for plan_id in ids if plan_id >= len(statements)]
    if missing:
        raise ValueError("Missing SQL templates for IDs {} in {}".format(
            missing[:10], sql_file))
    return [sql_template_signature(statements[plan_id]) for plan_id in ids]



def train_and_save_model_from_file(folder, train_file, adaptation_file, train_label_file, adaptation_label_file, verbose=True,
                                    emphasize_experiments=0, alpha=0.25, repeat=10, contrastive=False, meta=False, lr=0.001,
                                    adaptation_indices=None, epochs=100, batch_size=256, seed=0,
                                    template_contrastive=False, train_sql_file=None,
                                    adaptation_sql_file=None, static_target_weight=None,
                                    initial_model_folder=None):
    train_experience, train_labels, train_ids, train_alignment = load_aligned_experience(
        train_file, train_label_file)
    adaptation_expericence, adaptation_labels, adaptation_ids, adaptation_alignment = load_aligned_experience(
        adaptation_file, adaptation_label_file, adaptation_indices)
    all_experience = list(train_experience)
    all_labels = list(train_labels)
    sample_weights = ([1.0] * len(train_experience)
                      if static_target_weight is not None else None)
    template_signatures = None
    if template_contrastive:
        if not train_sql_file or not adaptation_sql_file:
            raise ValueError("Template contrastive learning requires SQL files")
        source_templates = load_sql_template_signatures(train_sql_file, train_ids)
        target_templates = load_sql_template_signatures(
            adaptation_sql_file, adaptation_ids)
        template_signatures = list(source_templates)
    all_experience.extend(adaptation_expericence)
    all_labels.extend(adaptation_labels)
    if sample_weights is not None:
        sample_weights.extend([float(static_target_weight)] * len(adaptation_labels))
    if template_signatures is not None:
        template_signatures.extend(target_templates)
    for i in range(repeat):
        all_experience.extend(adaptation_expericence)
        all_labels.extend(adaptation_labels)
        if sample_weights is not None:
            sample_weights.extend([float(static_target_weight)] * len(adaptation_labels))
        if template_signatures is not None:
            template_signatures.extend(target_templates)
     
    #for _ in range(emphasize_experiments):
    #    all_experience.extend(storage.experiment_experience())
    x = [i[0] for i in all_experience]
    #y = [i[1] for i in all_experience]        
    #y = [json.loads(i[0])["Plan"]["Total Cost"] for i in all_experience]
    y = all_labels

    if not all_experience:
        raise BaoTrainingException("Cannot train a Bao model with no experience")
    
    if len(all_experience) < 20:
        print("Warning: trying to train a Bao model with fewer than 20 datapoints.")

    template_labels = None
    if template_signatures is not None:
        template_map = {}
        template_labels = []
        for signature in template_signatures:
            template_labels.append(template_map.setdefault(signature, len(template_map)))
    reg = model.BaoRegression(have_cache_data=True, verbose=verbose, alpha=alpha, contrastive=contrastive, meta=meta, lr=lr,
                              template_contrastive=template_contrastive)
    if initial_model_folder is not None:
        reg.load(os.path.join(initial_model_folder, "model"), repeat)
    reg.fit(x, y, len(all_experience) - len(adaptation_expericence),
            epochs=epochs, batch_size=batch_size, seed=seed,
            template_labels=template_labels, sample_weights=sample_weights,
            warm_start=initial_model_folder is not None)
    save_path = folder + "/model"
    reg.save(save_path, repeat)

    train_x = x[:len(all_experience) - len(adaptation_expericence)]
    train_y = y[:len(all_experience) - len(adaptation_expericence)]
    predict = reg.predict(train_x)
    predict = [i[0] for i in predict]
    print("Training workload error:")
    qerror = print_qerror(predict, train_y)

    test_x = x[len(all_experience) - len(adaptation_expericence):]
    test_y = y[len(all_experience) - len(adaptation_expericence):]
    predict = reg.predict(test_x)
    predict = [i[0] for i in predict]
    print("Test workload error:")
    qerror = print_qerror(predict, test_y)
    return reg, {"train": train_alignment, "adaptation": adaptation_alignment}

def print_qerror(preds, labels):
    qerror = []
    for i in range(len(preds)):
        if preds[i] > float(labels[i]):
            qerror.append(preds[i] / float(labels[i]))
        else:
            qerror.append(float(labels[i]) / float(preds[i]))

    print("Median: {}".format(np.median(qerror)))
    #print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))
    return qerror

def predict(folder, prediction_file, prediction_label_file, alpha=0.25, repeat=10,
            contrastive=False, meta=False, lr=0.001, output_dir=None,
            template_contrastive=False, prediction_indices=None):
    predict_experience, predict_labels, plan_ids, alignment = load_aligned_experience(
        prediction_file, prediction_label_file, prediction_indices)
    x = [i[0] for i in predict_experience]
    #y = [i[1] for i in predict_experience]        
    #y = [json.loads(i[0])["Plan"]["Total Cost"] for i in predict_experience]
    y = predict_labels
    reg = model.BaoRegression(have_cache_data=True, alpha=alpha, contrastive=contrastive, meta=meta, lr=lr,
                              template_contrastive=template_contrastive)
    save_path = folder + "/model"
    reg.load(save_path, repeat)
    predict = reg.predict(x)
    predict = [i[0] for i in predict]
    print("Pridiction workload error:")
    qerror = print_qerror(predict, y)
    metrics = {
        "mean_qerror": float(np.mean(qerror)),
        "median_qerror": float(np.median(qerror)),
        "p90_qerror": float(np.percentile(qerror, 90)),
        "p95_qerror": float(np.percentile(qerror, 95)),
        "p99_qerror": float(np.percentile(qerror, 99)),
        "max_qerror": float(np.max(qerror)),
        "evaluation_count": len(qerror),
    }
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as stream:
            json.dump(metrics, stream, indent=2, sort_keys=True)
        with open(os.path.join(output_dir, "predictions.csv"), "w") as stream:
            stream.write("plan_id,label,prediction,qerror\n")
            for plan_id, label, prediction, error in zip(
                    plan_ids, y, predict, qerror):
                stream.write("{},{},{},{}\n".format(
                    plan_id, label, prediction, error))
        with open(os.path.join(output_dir, "evaluation_alignment.json"), "w") as stream:
            json.dump(alignment, stream, indent=2, sort_keys=True)
    else:
        np.savetxt("result.txt", qerror)
    nni.report_final_result(np.median(qerror))
    return metrics, alignment




if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: train.py MODEL_FILE")
        exit(-1)
    train_and_save_model(sys.argv[1], alpha=0.25)

    print("Model saved, attempting load...")
    reg = model.BaoRegression(have_cache_data=True, alpha=0.25)
    reg.load(sys.argv[1])
