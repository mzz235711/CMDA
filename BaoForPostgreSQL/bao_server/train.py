import storage
import model
import os
import shutil
import reg_blocker
import json
import nni

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



def train_and_save_model_from_file(folder, train_file, adaptation_file, train_label_file, adaptation_label_file, verbose=True,
                                    emphasize_experiments=0, alpha=0.25, repeat=10, contrastive=False, meta=False, lr=0.001):
    train_experience = load_experience(train_file)
    adaptation_expericence = load_experience(adaptation_file)
    train_labels = load_labels(train_label_file)
    adaptation_labels = load_labels(adaptation_label_file)
    all_experience = train_experience
    all_labels = train_labels
    all_experience.extend(adaptation_expericence)
    all_labels.extend(adaptation_labels)
    for i in range(repeat):
        all_experience.extend(adaptation_expericence)
        all_labels.extend(adaptation_labels)
     
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

    reg = model.BaoRegression(have_cache_data=True, verbose=verbose, alpha=alpha, contrastive=contrastive, meta=meta, lr=lr)
    reg.fit(x, y, len(all_experience) - len(adaptation_expericence))
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
    return reg

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

def predict(folder, prediction_file, prediction_label_file, alpha=0.25, repeat=10, contrastive=False, meta=False, lr=0.001):
    predict_experience = load_experience(prediction_file)
    predict_labels = load_labels(prediction_label_file)
    x = [i[0] for i in predict_experience]
    #y = [i[1] for i in predict_experience]        
    #y = [json.loads(i[0])["Plan"]["Total Cost"] for i in predict_experience]
    y = predict_labels
    reg = model.BaoRegression(have_cache_data=True, alpha=alpha, contrastive=contrastive, meta=meta, lr=lr)
    save_path = folder + "/model"
    reg.load(save_path, repeat)
    predict = reg.predict(x)
    predict = [i[0] for i in predict]
    print("Pridiction workload error:")
    qerror = print_qerror(predict, y)
    np.savetxt("result.txt", qerror)
    nni.report_final_result(np.median(qerror))




if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: train.py MODEL_FILE")
        exit(-1)
    train_and_save_model(sys.argv[1], alpha=0.25)

    print("Model saved, attempting load...")
    reg = model.BaoRegression(have_cache_data=True, alpha=0.25)
    reg.load(sys.argv[1])

