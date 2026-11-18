import os
from pathlib import Path

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


def load_job_ceb_dataset():
    """JOB template-drift workload prepared by the ResCard MSCN pipeline."""
    save_folder = os.environ.get("CMDA_JOB_DATA", str(Path(__file__).resolve().parent / "workloads/job"))
    train_query_file = save_folder + "/train.csv"
    predict_query_file = save_folder + "/test.csv"
    train_bitmap_file = save_folder + "/train_bitmap.csv"
    predicate_bitmap_file = save_folder + "/test_bitmap.csv"
    result_file = save_folder + "/result.csv"
    # Seeded training replaces these adaptation paths with the target pool.
    adaptation_file = predict_query_file
    adaptation_bitmap_file = predicate_bitmap_file
    validation_file = predict_query_file
    return (train_query_file, save_folder + "/", predict_query_file,
            result_file, save_folder, adaptation_file, validation_file,
            train_bitmap_file, predicate_bitmap_file,
            adaptation_bitmap_file, [])


def load_dsb_dataset():
    """DSB template-drift workload from ResCard with matching MSCN bitmaps."""
    query_root = os.environ.get("CMDA_DSB_QUERIES", str(Path(__file__).resolve().parent / "workloads/dsb"))
    bitmap_root = os.environ.get("CMDA_DSB_BITMAPS", query_root)
    train_query_file = query_root + "/dsb_train.csv"
    predict_query_file = query_root + "/dsb_test.csv"
    train_bitmap_file = bitmap_root + "/dsb_train_bitmap.csv"
    predicate_bitmap_file = bitmap_root + "/dsb_test_bitmap.csv"
    result_file = query_root + "/result.csv"
    adaptation_file = predict_query_file
    adaptation_bitmap_file = predicate_bitmap_file
    validation_file = predict_query_file
    return (train_query_file, bitmap_root + "/", predict_query_file,
            result_file, query_root, adaptation_file, validation_file,
            train_bitmap_file, predicate_bitmap_file,
            adaptation_bitmap_file, [])



