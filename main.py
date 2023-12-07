# File:                     main.py
# Created by:               Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2022/3/15
#
# Main file for the implementation on Biased-MNIST of the method presented in
# "Mining Bias-Target Alignement from Voronoi Cells"
# ================================= IMPORTS =================================
from utils.configs import *
from utils.tools import *
from utils.vcba.basemodel import VCBABaseModel
from utils.vcba.train_and_debias import *
import torch
import os
import wandb

# ================================== CODE ===================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main_vcba(args):
    (dataset_params,
     biased_learning_params,
     debiased_learning_params,
     ph_learning_params) = get_params_dict(args)
    dataloaders = build_dataloaders(dataset_params)
    log_action("Debiased model initialization...", importance=2)
    debiased_model = VCBABaseModel(
        device=args.device,
        learning_params=debiased_learning_params,
        dataset_params=dataset_params,
    )
    debiased_model.set_data_loaders(dataloaders)
    debiased_model.set_information_removal(
        gamma=args.gamma, ph_learning_params=ph_learning_params
    )
    if not args.bias_supervised:
        log_action("Biased model initialization...", importance=2)
        biased_model = VCBABaseModel(
            learning_params=biased_learning_params,
            dataset_params=dataset_params,
            device=args.device,
        )
        biased_model.set_data_loaders(dataloaders)

        if not os.path.exists(args.epoch_stats_file):
            log_action("Starting to infer bias alignment...")
            infer_predicted_bias(
                biased_base_model=biased_model,
                nb_prep_epochs=args.warmup_epochs,
                metric=biased_model.reweighter.bias_inference_metric,
            )
            (rho_per_class, pb_labels, _) = rho_class(biased_model=biased_model)
            print("here")
        else:
            log_action(
                f"Retrieving bias alignement info from {args.epoch_stats_file}..."
            )
            (rho_per_class, pb_labels, _) = rho_class(
                biased_model=biased_model,
                stats_dist_file_per_epoch=args.epoch_stats_file,
            )
        debiased_model.reweighter.set_predicted_bias_labels(
            predicted_bias_target=pb_labels, predicted_rho=rho_per_class
        )
        predicted_bias_labels = build_predicted_bias_labels(
            predicted_bias_labels=pb_labels,
            train_loader=debiased_model.dataloaders["train"],
            device=debiased_model.device,
            num_classes=debiased_model.nb_classes,
        )
    else:
        supervised_rho = debiased_model.reweighter.get_rho_from_bias_labels(
            debiased_model.dataloaders["train"]
        )
        print(supervised_rho)
        debiased_model.reweighter.set_supervised(predicted_rho=supervised_rho)
        predicted_bias_labels = None
    # Debiased model training
    log_action("Training the debiased model", importance=2)
    for epoch in range(1, args.epochs + 1):
        train_test_epoch(
            base_model=debiased_model,
            epoch=epoch,
            predicted_bias_labels=predicted_bias_labels,
        )
    
    torch.save(
        debiased_model.model.state_dict(),
        f"data/models/Bmnist_{args.rho}_LR{args.lr}_BiasLR{args.bias_lr}_GAMMA{args.gamma}_SEED{args.seed}.pth",
    )


def main(args):
    log_action(
        f"Launching the debiasing for Biased-MNIST with rho = {args.rho}", importance=1
    )
    set_device(args)
    set_seeds(args.seed)
    if args.method == "vcba":
        main_vcba(args)


if __name__ == "__main__":
    if args.log:
        wandb.init(config={}, project=args.project)
        args = wandb.config
    main(args)
