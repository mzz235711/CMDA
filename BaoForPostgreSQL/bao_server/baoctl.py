import argparse
import socket
import json
import time
import dataset
import os
try:
    import nni
except Exception:
    class _NniFallback:
        @staticmethod
        def get_next_parameter():
            return {}
    nni = _NniFallback()

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
    parser.add_argument("--template_contrastive", action="store_true",
                        help="use query-template supervised contrastive learning")
    parser.add_argument("--static_target_weight", type=float,
                        help="fixed per-target weight; enables Static-Upweight")
    parser.add_argument("--static_train_indices")
    parser.add_argument("--static_validation_indices")
    parser.add_argument("--meta", action="store_true", help='use meta learning')
    parser.add_argument("--repeat", type=int, default=10, help='hyperparameter')
    parser.add_argument("--alpha", type=float, default=0.25, help='hyperparameter')
    parser.add_argument("--lr", type=float, default=0.001, help='hyperparameter')
    parser.add_argument("--seed", type=int)
    parser.add_argument("--adaptation_budget", type=int, default=10)
    parser.add_argument("--output_root", default="../../results/cost_multiseed")
    parser.add_argument("--method_name")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--prepare_selection_only", action="store_true")
    parser.add_argument("--selection_indices",
                        help="explicit comma-separated adaptation IDs selected by an external policy")
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

    if args.seed is not None:
        import cost_experiment
        import train
        config = dataset.load_multiseed_dataset(args.dataset)
        output_root = os.path.abspath(args.output_root)
        selection, selection_file = cost_experiment.load_or_create_selection(
            output_root, args.dataset, args.adaptation_budget, args.seed,
            config["adaptation_plans"], config["adaptation_labels"])
        if args.selection_indices:
            explicit = sorted(int(value) for value in args.selection_indices.split(",") if value)
            pool_ids = set(train.load_aligned_experience(
                config["adaptation_plans"], config["adaptation_labels"])[2])
            if (len(explicit) != args.adaptation_budget or
                    len(set(explicit)) != len(explicit) or
                    not set(explicit).issubset(pool_ids)):
                raise ValueError("Explicit selection must contain adaptation_budget unique pool IDs")
            selection = {**selection, "adaptation_indices": explicit,
                         "selection_id": cost_experiment.ids_hash(explicit),
                         "selection_policy": "external"}
            external_dir = os.path.join(output_root, args.dataset, "external_selections")
            os.makedirs(external_dir, exist_ok=True)
            selection_file = os.path.join(
                external_dir, "{}_seed_{:03d}.json".format(
                    args.method_name or "external", args.seed))
            with open(selection_file, "w") as stream:
                json.dump(selection, stream, indent=2, sort_keys=True)
        full_adaptation_indices = selection["adaptation_indices"]
        static_train_indices = ([int(x) for x in args.static_train_indices.split(",") if x]
                                if args.static_train_indices else None)
        static_validation_indices = ([int(x) for x in args.static_validation_indices.split(",") if x]
                                     if args.static_validation_indices else None)
        if static_train_indices is not None:
            if args.static_target_weight is None or static_validation_indices is None:
                raise ValueError("Static inner CV requires weight, train IDs, and validation IDs")
            if (set(static_train_indices) | set(static_validation_indices) != set(full_adaptation_indices) or
                    set(static_train_indices) & set(static_validation_indices)):
                raise ValueError("Static CV IDs must partition adaptation IDs")
            selection = {**selection, "adaptation_indices": static_train_indices}
        evaluation_id, evaluation_alignment = cost_experiment.evaluation_identity(
            config["evaluation_plans"], config["evaluation_labels"])
        method_name = args.method_name or (
            "cmda" if args.contrastive and args.meta else
            "cmda_c" if args.contrastive else
            "cmda_m" if args.meta else "adapt")
        run_dir = os.path.join(
            output_root, args.dataset,
            "budget_{}".format(args.adaptation_budget), method_name,
            "seed_{:03d}".format(args.seed))
        os.makedirs(run_dir, exist_ok=True)
        manifest = {
            **selection,
            "method": method_name,
            "contrastive": args.contrastive,
            "template_contrastive": args.template_contrastive,
            "static_target_weight": args.static_target_weight,
            "full_adaptation_indices": full_adaptation_indices,
            "static_validation_indices": static_validation_indices,
            "meta": args.meta,
            "alpha": params["alpha"],
            "learning_rate": params["lr"],
            "repeat": params["repeat"],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "selection_manifest": os.path.abspath(selection_file),
            "evaluation_id": evaluation_id,
            "evaluation_alignment": evaluation_alignment,
        }
        with open(os.path.join(run_dir, "manifest.json"), "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
        print("Adaptation query IDs:", selection["adaptation_indices"])
        print("Selection ID:", selection["selection_id"])
        print("Evaluation ID:", evaluation_id)
        if args.prepare_selection_only:
            raise SystemExit(0)
        cost_experiment.set_global_seed(args.seed)
        start_time = time.time()
        _, alignments = train.train_and_save_model_from_file(
            run_dir, config["train_plans"], config["adaptation_plans"],
            config["train_labels"], config["adaptation_labels"],
            contrastive=args.contrastive, meta=args.meta,
            repeat=params["repeat"], alpha=params["alpha"], lr=params["lr"],
            adaptation_indices=selection["adaptation_indices"],
            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
            template_contrastive=args.template_contrastive,
            train_sql_file=config["train_sql"],
            adaptation_sql_file=config["adaptation_sql"],
            static_target_weight=args.static_target_weight)
        manifest["training_alignment"] = alignments
        manifest["training_seconds"] = time.time() - start_time
        with open(os.path.join(run_dir, "manifest.json"), "w") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
        prediction_plans = (config["adaptation_plans"] if static_validation_indices is not None
                            else config["evaluation_plans"])
        prediction_labels = (config["adaptation_labels"] if static_validation_indices is not None
                             else config["evaluation_labels"])
        train.predict(
            run_dir, prediction_plans, prediction_labels,
            alpha=params["alpha"], repeat=params["repeat"],
            contrastive=args.contrastive, meta=args.meta, lr=params["lr"],
            output_dir=run_dir,
            template_contrastive=args.template_contrastive,
            prediction_indices=static_validation_indices)
        raise SystemExit(0)

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


    
