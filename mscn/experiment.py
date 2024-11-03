from nni.experiment import Experiment
import os

search_space = {
    #'lr': {'_type': 'loguniform', '_value': [0.001, 0.01]},
    #'alpha': {'_type': 'uniform', '_value': [0, 1]},
    'repeat': {'_type': 'randint', '_value': [0, 50]},
}

experiment = Experiment('local')
experiment.config.trial_command = 'CUDA_VISIBLE_DEVICES=7 python3 -u run.py --dataset stats --query_num 5000 --train_mode train --alpha 0.5 --epochs 50 --device cuda --num_samples 1000 --contrastive --meta'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.max_trial_number = 10 
experiment.config.trial_concurrency = 1 
experiment.start(8084)
input("press any key to exit")
