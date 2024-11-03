def load_job_dataset():
    save_folder = "./workloads/job-light-ranges"
    train_query_file = save_folder + "/joblight_train_5000.csv"
    min_max_file = save_folder + "/" 
    predict_query_file = save_folder + "/adaptation_2_prediction.csv"
    result_file = save_folder + "/result.csv"
    adaptation_file = save_folder + "/adaptation_10_adaptation.csv"
    validation_file = save_folder + "/validation.csv"
    train_bitmap_file = save_folder + "/joblight_train_5000_bitmap.csv"
    predicate_bitmap_file = save_folder + "/adaptation_2_prediction_bitmap.csv"
    adaptation_bitmap_file = save_folder + "/adaptation_10_adaptation_bitmap.csv"
    tables = ['cast_info', 'movie_companies', 'movie_info_idx', 'movie_info', 'movie_keyword', 'title']
    return train_query_file, min_max_file, predict_query_file, result_file, save_folder, \
            adaptation_file, validation_file, train_bitmap_file, predicate_bitmap_file, adaptation_bitmap_file, tables

def load_stats_dataset():
    save_folder = "./workloads/stats"
    train_query_file = save_folder + "/source_train.csv"
    min_max_file = save_folder + "/" 
    predict_query_file = save_folder + "/target_prediction.csv"
    result_file = save_folder + "/result.csv"
    adaptation_file = save_folder + "/adaptation_10.csv"
    validation_file = save_folder + "/validation.csv"
    train_bitmap_file = save_folder + "/source_train_bitmap.csv"
    predicate_bitmap_file = save_folder + "/target_prediction_bitmap.csv"
    adaptation_bitmap_file = save_folder + "/adaptation_10_bitmap.csv"
    tables = ['cast_info', 'movie_companies', 'movie_info_idx', 'movie_info', 'movie_keyword', 'title']
    return train_query_file, min_max_file, predict_query_file, result_file, save_folder, \
            adaptation_file, validation_file, train_bitmap_file, predicate_bitmap_file, adaptation_bitmap_file, tables



