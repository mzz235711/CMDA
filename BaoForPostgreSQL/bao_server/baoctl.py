import argparse
import socket
import json
import time
import dataset
import nni

def __json_bytes(obj):
    return (json.dumps(obj) + "\n").encode("UTF-8")

def __connect():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("localhost", 9381))
    return s

def send_model_load(path):
    with __connect() as s:
        s.sendall(__json_bytes({"type": "load model"}))
        s.sendall(__json_bytes({"path": path}))
        s.sendall(__json_bytes({"final": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Bao for PostgreSQL Controller")
    parser.add_argument("--dataset", type=str, help="Dataset")
    parser.add_argument("--train_mode", type=str, default="train", help="train, adaptation or predict")
    parser.add_argument("--contrastive", action='store_true', help='use contrastive learning')
    parser.add_argument("--meta", action="store_true", help='use meta learning')
    parser.add_argument("--repeat", type=int, default=10, help='hyperparameter')
    parser.add_argument("--alpha", type=float, default=0.25, help='hyperparameter')
    parser.add_argument("--lr", type=float, default=0.001, help='hyperparameter')
    parser.add_argument("--load",
                        metavar="PATH",
                        help="Load the saved Bao model")
    parser.add_argument("--train",
                        metavar="PATH",
                        help="Train a Bao model and save it")
    parser.add_argument("--retrain", action="store_true",
                        help="Force the Bao server to train a model and load it")
    parser.add_argument("--test-connection", action="store_true",
                        help="Test the connection from the Bao server to the PostgreSQL instance.")
    parser.add_argument("--add-test-query", metavar="PATH",
                        help="Add the SQL query in the file at PATH to the test query list")
    parser.add_argument("--status", action="store_true",
                        help="Print out information about the Bao server.")
    parser.add_argument("--experiment", metavar="SECONDS", type=int,
                        help="Conduct experiments on test queries for (up to) SECONDS seconds.")
    
    args = parser.parse_args()
    params = {
        'lr': args.lr,
        'alpha': args.alpha,
        'repeat': args.repeat,
    }
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)

    if args.dataset == "test":
        train_file, predict_file, save_folder, adaptation_file =  dataset.load_test_dataset()
    elif args.dataset == "job-light-ranges":
        train_file, predict_file, save_folder, adaptation_file, train_label_file, predict_label_file, adaptation_label_file = dataset.load_job_light_dataset()
    elif args.dataset == "job":
        train_file, predict_file, save_folder, adaptation_file, train_label_file, predict_label_file, adaptation_label_file = dataset.load_job_dataset()
    elif args.dataset == "dsb":
        train_file, predict_file, save_folder, adaptation_file, train_label_file, predict_label_file, adaptation_label_file = dataset.load_dsb_dataset()
    elif args.dataset == 'stats':
        train_file, predict_file, save_folder, adaptation_file, train_label_file, predict_label_file, adaptation_label_file = dataset.load_stats_dataset()


    if args.train_mode == 'train':
        import train
        print("Initial training from collected experience")
        start_time = time.time()
        train.train_and_save_model_from_file(save_folder, train_file, adaptation_file, train_label_file, adaptation_label_file, contrastive=args.contrastive,
                                             meta=args.meta, repeat=params['repeat'], alpha=params['alpha'], lr=params['lr'])
        end_time = time.time()
        print("Total training time: {}s".format(end_time - start_time))
        print("")
        print("Prediction")
        train.predict(save_folder, predict_file, predict_label_file, alpha=params['alpha'], repeat=params['repeat'], contrastive=args.contrastive,
                    meta=args.meta, lr=params['lr'])
    
    elif args.train_mode == 'predict':
        import train
        print("Prediction")
        train.predict(save_folder, predict_file, predict_label_file, alpha=params['alpha'], repeat=params['repeat'], contrastive=args.contrastive,
                    meta=args.meta, lr=params['lr'])



    if args.train:
        import train
        print("Training Bao model from collected experience")
        train.train_and_save_model(args.train)
        exit(0)

    if args.load:
        import model
        print("Attempting to load the Bao model...")
        reg = model.BaoRegression(have_cache_data=True)
        reg.load(args.load)
        
        print("Model loaded. Sending message to Bao server...")
        send_model_load(args.load)
        print("Message sent to server.")
        exit(0)

    if args.retrain:
        import train
        from constants import DEFAULT_MODEL_PATH, OLD_MODEL_PATH, TMP_MODEL_PATH
        train.train_and_swap(DEFAULT_MODEL_PATH, OLD_MODEL_PATH, TMP_MODEL_PATH,
                             verbose=True)
        send_model_load(DEFAULT_MODEL_PATH)
        exit(0)

    if args.test_connection:
        from reg_blocker import ExperimentRunner
        er = ExperimentRunner()
        if er.test_connection():
            print("Connection successful!")
            exit(0)
        else:
            print("Could not connect to PostgreSQL.")
            exit(1)
        
    if args.add_test_query:
        from reg_blocker import ExperimentRunner
        er = ExperimentRunner()

        with open(args.add_test_query) as f:
            sql = f.read()
        
        er.add_experimental_query(sql)
        exit(0)

    if args.experiment:
        from reg_blocker import ExperimentRunner
        er = ExperimentRunner()
        er.explore(args.experiment)
        exit(0)

    if args.status:
        from reg_blocker import ExperimentRunner
        er = ExperimentRunner()
        info = er.status()

        max_key_length = max(len(x) for x in info.keys())

        for k, v in info.items():
            print(k.ljust(max_key_length), ":", v)
            
        exit(0)


    
