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
