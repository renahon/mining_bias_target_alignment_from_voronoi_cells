# File:                     configs.py
# Created by:               Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2022/3/15
#
# Defines the arguments used in the main.py file
# ================================= IMPORTS =================================
import argparse
import os

import wandb

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ================================== CODE ===================================
parser = argparse.ArgumentParser()
# ----------------------------------- GENERAL -----------------------------------#
parser.add_argument("--dev", default="cuda:0", type=str)
parser.add_argument("--seed", type=int, default=47)
parser.add_argument("--batch_size", default=100, type=int)
parser.add_argument("--test_batch_size", default=100, type=int)
parser.add_argument("--epochs", default=80, type=int)
parser.add_argument("--method", type=str, default="vcba")
# ------------------------------------ WANDB ------------------------------------ #
parser.add_argument("--log", type=bool, default=False)
parser.add_argument("--project", type=str, default="VCBA_dev")
# ----------------------------------- DATASETS -----------------------------------#
parser.add_argument("--datapath", type=str, default="./data/")
# Biased-MNIST
parser.add_argument(
    "--rho",
    type=float,
    default=0.99,
    help="digit-color correlation level for biased mnist (.999, .997, .995, .990)",
)
# ------------------------------------- VCBA -------------------------------------#
parser.add_argument("--bias_supervised", type=bool, default=False)
####============================== BIASED MODEL ==============================####
# TRAINING PARAMETERS
parser.add_argument("--bias_lr", type=float, default=None)
parser.add_argument("--bias_momentum_sgd", type=float, default=None)
parser.add_argument("--bias_weight_decay", type=float, default=None)
parser.add_argument("-m",
                    "--bias_inference_metric",
                    type=str,
                    default="voronoi",
                    choices=["voronoi"],
                    )
# HYPERPLANE DISTANCE
parser.add_argument(
    "--warmup_epochs",
    type=int,
    default=1,
    help="Number of epochs without measuring the hyperplane distance",
)
parser.add_argument("--save_hyp_dist", type=str, default="true")
####============================= DEBIASED MODEL =============================####
# TRAINING PARAMETERS
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--momentum_sgd", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
# MUTUAL INFORMATION REMOVAL
parser.add_argument("--gamma", default=2, type=float)
# PRIVACY HEAD TRAINING PARAMETERS
parser.add_argument("--ph_lr", type=float, default=None)
parser.add_argument("--ph_momentum_sgd", type=float, default=None)
parser.add_argument("--ph_weight_decay", type=float, default=None)
# --------------------------------- SAVE AND LOG ---------------------------------#
args = parser.parse_args()
# --------------------------------- UPDATE ARGS ----------------------------------#
if args.ph_lr is None:
    args.ph_lr = args.lr
if args.bias_lr is None:
    args.bias_lr = args.lr
if args.ph_momentum_sgd is None:
    args.ph_momentum_sgd = args.momentum_sgd
if args.bias_momentum_sgd is None:
    args.bias_momentum_sgd = args.momentum_sgd
if args.ph_weight_decay is None:
    args.ph_weight_decay = args.weight_decay
if args.bias_weight_decay is None:
    args.bias_weight_decay = args.weight_decay
args.voronoi_dist_file = (
    f"data/stats_files/voronoi_dist_S{args.seed}_RHO{args.rho}_LR{args.bias_lr}.pth"
)
args.epoch_stats_file = (
    f"data/stats_files/epoch_stats_D{args.bias_inference_metric}_S{args.seed}_RHO{args.rho}_LR{args.bias_lr}.pth"
)


def get_dataset_params(args):
    dataset_params = {
        "dataset": "Bmnist",
        "dataset_size": 60000,
        "datapath": args.datapath,
        "rho": args.rho,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
    }
    return dataset_params


def get_learning_params(args, model_role: str = "debiased"):
    model_roles = ["debiased", "biased", "ph"]
    if model_role not in model_roles:
        raise ValueError(
            "Invalid model_role. Expected one of: %s" % model_roles)
    if model_role == "debiased":
        learning_params = {
            "role": "debiased",
            "lr": args.lr,
            "momentum_sgd": args.momentum_sgd,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "epochs": args.epochs,
            "criterion_reduction": "none",
            "reweigh": True,
            "bias_inference_metric": None,
        }
    elif model_role == "biased":
        learning_params = {
            "role": "biased",
            "lr": args.bias_lr,
            "momentum_sgd": args.bias_momentum_sgd,
            "weight_decay": args.bias_weight_decay,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "warmup_epochs": args.warmup_epochs,
            "epochs": args.epochs,
            "criterion_reduction": "none",
            "reweigh": False,
            "bias_inference_metric": args.bias_inference_metric,
        }
    elif model_role == "ph":
        learning_params = {
            "role": "ph",
            "lr": args.ph_lr,
            "momentum_sgd": args.ph_momentum_sgd,
            "weight_decay": args.ph_weight_decay,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "epochs": args.epochs,
            "criterion_reduction": "none",
            "reweigh": False,
            "bias_inference_metric": None,
        }
    #print(learning_params)
    return learning_params


def get_params_dict(conf):
    dataset_params = get_dataset_params(conf)
    biased_learning_params = get_learning_params(conf, "biased")
    debiased_learning_params = get_learning_params(conf, "debiased")
    ph_learning_params = get_learning_params(conf, "ph")
    return (
        dataset_params,
        biased_learning_params,
        debiased_learning_params,
        ph_learning_params,
    )
