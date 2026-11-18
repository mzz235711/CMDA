def load_job_light_dataset():
    save_folder = "./workloads/job-light-ranges"
    train_query_file = save_folder + "/joblight_train_5000_plans"
    train_label_file = save_folder + "/joblight_train_5000_time.txt"
    predict_query_file = save_folder + "/job-light_70_plans"
    predict_label_file = save_folder + "/job-light_70_time.txt"
    adaptation_query_file = save_folder + "/adaptation_10_adaptation_plans"
    adaptation_label_file = save_folder + "/adaptation_10_adaptation_time.txt"
    return train_query_file, predict_query_file, save_folder, adaptation_query_file, train_label_file, predict_label_file, adaptation_label_file

def load_stats_dataset():
    save_folder = "./workloads/stats"
    train_query_file = save_folder + "/source_train_only_plans"
    train_label_file = save_folder + "/source_train_only_time.txt"
    predict_query_file = save_folder + "/target_prediction_only_plans"
    predict_label_file = save_folder + "/target_prediction_only_time.txt"
    adaptation_query_file = save_folder + "/adaptation_10_only_plans"
    adaptation_label_file = save_folder + "/adaptation_10_only_time.txt"
    return train_query_file, predict_query_file, save_folder, adaptation_query_file, train_label_file, predict_label_file, adaptation_label_file


def load_multiseed_dataset(name):
    """Return the fixed source/evaluation sets and full adaptation pool."""
    if name == "job-light-ranges":
        base = "./workloads/job-light-ranges"
        return {
            "save_folder": base,
            "train_plans": base + "/joblight_train_5000_plans",
            "train_labels": base + "/joblight_train_5000_time.txt",
            "train_sql": base + "/joblight_train_5000.sql",
            "adaptation_plans": base + "/adaptation_all_train_plans",
            "adaptation_labels": base + "/adaptation_all_train_time.txt",
            "adaptation_sql": base + "/adaptation_all_train.sql",
            "evaluation_plans": base + "/job-light_70_plans",
            "evaluation_labels": base + "/job-light_70_time.txt",
            "evaluation_sql": base + "/job-light_70.sql",
        }
    if name == "stats":
        base = "./workloads/stats"
        return {
            "save_folder": base,
            "train_plans": base + "/source_train_only_plans",
            "train_labels": base + "/source_train_only_time.txt",
            "train_sql": base + "/source_train_only.sql",
            "adaptation_plans": base + "/adaptation_all_only_plans",
            "adaptation_labels": base + "/adaptation_all_only_time.txt",
            "adaptation_sql": base + "/adaptation_all_only.sql",
            "evaluation_plans": base + "/target_prediction_only_plans",
            "evaluation_labels": base + "/target_prediction_only_time.txt",
            "evaluation_sql": base + "/target_prediction_only.sql",
        }
    if name in ("job", "dsb"):
        base = "./workloads/{}".format(name)
        return {
            "save_folder": base,
            "train_plans": base + "/source_plans",
            "train_labels": base + "/source_cost.txt",
            "train_sql": base + "/source.sql",
            "adaptation_plans": base + "/target_plans",
            "adaptation_labels": base + "/target_cost.txt",
            "adaptation_sql": base + "/target.sql",
            "evaluation_plans": base + "/target_plans",
            "evaluation_labels": base + "/target_cost.txt",
            "evaluation_sql": base + "/target.sql",
        }
    raise ValueError(
        "Multi-seed cost experiments support job-light-ranges, stats, job, and dsb")
