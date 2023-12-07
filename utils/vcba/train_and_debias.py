# File:                     train_and_debias.py
# Created by:               Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2022/3/15
#
# ================================= IMPORTS =================================
from torch.nn.functional import one_hot
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from utils.configs import *
from utils.tools import print_remi
from utils.vcba.basemodel import VCBABaseModel
from utils.tools import *

# ================================== CODE ===================================
__all__ = [
    "compute_weighted_loss",
    "get_centroids_from_batch",
    "infer_predicted_bias",
    "train_biased_model",
    "train_test_epoch",
    "rho_class",
    "train_with_information_removal",
    "test_with_information_removal",
    "count_elements_per_class",
    "build_predicted_bias_labels",
    "gather_stats"
]
SAMPLES_TYPES = {"misclassified": "Misclassified",
                 "well_classified": "Correctly Classified",
                 "WCA": "ỹ=y, y=b, ỹ=b",
                 "WnCnA": "ỹ=y, y≠b, ỹ≠b",
                 "nWnCA": "ỹ≠y, y=b, ỹ≠b",
                 "nWCnA": "ỹ≠y, y≠b, ỹ=b",
                 "nWnCnA": "ỹ≠y, y≠b, ỹ≠b"}
METRICS_LIST = {"softmax_y_target": "softmax(y)",
                "softmax_y_out": "softmax(ỹ)",
                "d_correct_centroids": "D[x,c(y)]",
                "d_inference_centroids": "D[x,c(ỹ)]",
                "d_bias_centroids": "D[x,c(b)]",
                "d_hyperplane": "D[x,H(y,ỹ)]"}


def compute_weighted_loss(
    base_model: VCBABaseModel,
    output: torch.Tensor,
    output_bottleneck: torch.Tensor,
    target: torch.Tensor,
    indices: torch.Tensor = None,
    bias_target: torch.Tensor = None,
):
    base_model.reweighter.update(
        output_model=output,
        output_bottleneck=output_bottleneck,
        target=target,
        indices=indices,
        bias_target=bias_target,
        keep_centro=True,
    )
    if bias_target is not None:
        weights = base_model.reweighter.weigh(bias_target=bias_target)
    else:
        weights = base_model.reweighter.weigh()
    # print(weights)
    weighted_loss = torch.mean(weights * base_model.criterion(output, target))
    return weighted_loss


def get_centroids_from_batch(base_model: VCBABaseModel):
    """
    Computes the centroids, representative of each target class
    on the whole dataset at once, without updating the model.
    """
    with torch.no_grad():
        base_model.model.eval()
        base_model.reweighter.reset()
        tk0 = tqdm(
            base_model.dataloaders["train"],
            total=int(len(base_model.dataloaders["train"])),
            leave=True,
        )
        for _, (data, target, bias_target, _) in enumerate(tk0):
            data = data.to(base_model.device)
            target = target.to(base_model.device)
            output = base_model.model(data).to(base_model.device)
            bias_target = bias_target.to(base_model.device)
            output_bottleneck = base_model.bottleneck.output
            base_model.reweighter.update(
                output_model=output,
                output_bottleneck=output_bottleneck,
                target=target,
                bias_target=bias_target,  # careful : added for stats
                keep_centro=False,
            )


def init_stats():
    stats_dict = {}

    for s in SAMPLES_TYPES:
        for m in METRICS_LIST.keys():
            stats_dict[f"{s}_{m}"] = AverageMeter(f"{s}_{m}", ":.4f")
    return stats_dict


def get_wandbstats(stats_dict):
    wandb_stats_dict = {}
    for s in SAMPLES_TYPES.keys():
        wandb_stats_dict[f"{s} count"] = stats_dict[f"{s}_softmax_y_target"].count
        for m in METRICS_LIST.keys():
            wandb_stats_dict[f'{SAMPLES_TYPES[s]} - {METRICS_LIST[m]}'] = stats_dict[f"{s}_{m}"].avg
            wandb_stats_dict[f'{SAMPLES_TYPES[s]} - {METRICS_LIST[m]} - Standard Deviation'] = stats_dict[f"{s}_{m}"].std

    return wandb_stats_dict


def gather_stats(base_model: VCBABaseModel):

    for epoch in range(base_model.epochs + 1):
        print(f"Computing centroids for epoch {epoch}")
        get_centroids_from_batch(base_model)
        base_model.reweighter.define_hyperplanes()
        base_model.model.train()
        stats_dict = init_stats()
        loss_task_tot = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        tk0 = tqdm(
            base_model.dataloaders["train"],
            total=int(len(base_model.dataloaders["train"])),
            leave=True,
        )
        for _, (data, y_target, bias_target, idx) in enumerate(tk0):
            data = data.to(base_model.device)
            y_target = y_target.to(base_model.device)
            bias_target = bias_target.to(base_model.device)
            idx = idx.to(base_model.device)
            softmax_output = base_model.model(data).to(base_model.device)
            y_output = torch.argmax(softmax_output, dim=1)
            output_bottleneck = base_model.bottleneck.output
            base_model.reweighter.update(
                output_model=softmax_output,
                output_bottleneck=output_bottleneck,
                target=y_target,
                indices=idx,
                bias_target=bias_target,
                keep_centro=True)
            distances_from = base_model.reweighter.update_dist_lists(
                epoch, "stats")
            loss_task = torch.mean(
                base_model.criterion(softmax_output, y_target))
            loss_task_tot.update(loss_task.item(), data.size(0))
            loss_task.backward()
            base_model.optimizer.step()
            base_model.optimizer.zero_grad()
            acc1 = accuracy(softmax_output, y_target, topk=(1,))
            top1.update(acc1[0], data.size(0))
            # What do I want to get as stats from that epoch ?
            # - average and standard deviation of
            #       - relative distance to hyperplane
            #       - relative distance to correct centroids  : y_out = y_target
            #       - relative distance to inferred centroids : y_out
            #       - relative distance to bias centroids :     bias_target
            #       - ratio of distances to correct/inferred
            #       - softmax for
            #           - y_output
            #           - y_target
            # - number of samples
            #
            # for all combinations of
            #      - W: well-classified or not (nW):    y_out = y_target
            #      - A: bias-aligned or not    (nA): y_target = bias_target
            #      - C: bias-captured or not   (nC):    y_out = bias_target
            update_stats(stats_dict, distances_from, torch.softmax(softmax_output, dim=1),
                         y_target, y_output, bias_target)
        wandb_stats = get_wandbstats(stats_dict)
        wandb_stats['epoch'] = epoch
        wandb_stats['loss'] = loss_task_tot.avg
        wandb_stats['acc'] = top1.avg.item()
        wandb.log(wandb_stats)


def update_stats_configuration(name, mask, stats_dict, distances_from, softmax_output, y_target, y_output, bias_target, nb_classes):
    # Softmaxes
    softmax_masked = softmax_output*mask.unsqueeze(-1)
    s = softmax_masked*one_hot(y_target, nb_classes)
    s_masked = torch.masked_select(s, s > 0)
    nb_samples = s_masked.size(0)
    s_mean = torch.mean(s_masked)
    if not torch.isnan(s_mean).item():
        stats_dict[f"{name}_softmax_y_target"].update(
            s_mean.item(), n=nb_samples)
    s = softmax_masked*one_hot(y_output, nb_classes)
    s_masked = torch.masked_select(s, s > 0)
    nb_samples = s_masked.size(0)
    s_mean = torch.mean(s_masked)
    if not torch.isnan(s_mean).item():
        stats_dict[f"{name}_softmax_y_out"].update(
            s_mean.item(), n=nb_samples)
    # Distances
    dist_to_center = torch.mean(distances_from["centroids_center"]).item()
    for obj in ["correct_centroids", "inference_centroids", "bias_centroids", "hyperplane"]:
        # / torch.mean(distances_from["centroids_center"])
        d = distances_from[obj] * mask
        d_masked = torch.masked_select(d, d > 0)
        nb_samples = d_masked.size(0)
        d_mean = torch.mean(d_masked)
        if not torch.isnan(d_mean).item():
            if dist_to_center > 0:
                stats_dict[f"{name}_d_{obj}"].update(
                    d_mean.item()/dist_to_center,
                    n=nb_samples)
    # avg_dist_h = torch.sum(torch.nan_to_num(base_model.reweighter.stats_dist[:, epoch - 1, 0], nan=0.0)
    #                       ) / torch.sum(mask)
    # rel_dist = torch.sum(
    #    torch.nan_to_num(
    #        base_model.reweighter.stats_dist[:, epoch - 1, 2], nan=0.0)
    # ) / torch.sum(mask)
    # misclassified += (
    #    (y_target != torch.argmax(softmax_output, dim=1)).nonzero().size(0)
    # )
    # elts_tot = torch.sum((base_model.reweighter.dist_from_hyperplane[:, 0] != float('nan'))
    #                     ).item()
    # tk0.set_postfix(loss_task=loss_task_tot.avg,
    #                top1=top1.avg.item(),
    #                d_avg=avg_dist_h.item(),
    #                d_rel=rel_dist.item(),
    #                m_total=elts_tot,
    #                m_epoch=misclassified)
#
    #        return loss_task_tot.avg, top1.avg.item()


def update_stats(stats_dict, distances_from, softmax_output, y_target, y_output, bias_target, nb_classes=10):
    mask_W = y_output == y_target
    mask_A = y_target == bias_target
    mask_C = y_output == bias_target
    # WCA
    name = "WCA"
    mask = mask_W*mask_A*mask_C
    update_stats_configuration(name, mask, stats_dict, distances_from,
                               softmax_output, y_target, y_output, bias_target, nb_classes)
    # WnCnA
    name = "WnCnA"
    mask = mask_W*torch.logical_not(mask_A)*torch.logical_not(mask_C)
    update_stats_configuration(name, mask, stats_dict, distances_from,
                               softmax_output, y_target, y_output, bias_target, nb_classes)
    # nWnCA
    name = "nWnCA"
    mask = torch.logical_not(mask_W)*mask_A*torch.logical_not(mask_C)
    update_stats_configuration(name, mask, stats_dict, distances_from,
                               softmax_output, y_target, y_output, bias_target, nb_classes)
    # nWCnA
    name = "nWCnA"
    mask = torch.logical_not(mask_W)*torch.logical_not(mask_A)*mask_C
    update_stats_configuration(name, mask, stats_dict, distances_from,
                               softmax_output, y_target, y_output, bias_target, nb_classes)
    # nWnCnA
    name = "nWnCnA"
    mask = torch.logical_not(
        mask_W)*torch.logical_not(mask_A)*torch.logical_not(mask_C)
    update_stats_configuration(name, mask, stats_dict, distances_from,
                               softmax_output, y_target, y_output, bias_target, nb_classes)
    # Misclassified
    name = "misclassified"
    mask = torch.logical_not(mask_W)
    update_stats_configuration(name, mask, stats_dict, distances_from,
                               softmax_output, y_target, y_output, bias_target, nb_classes)
    # Well classified
    name = "well_classified"
    mask = mask_W
    update_stats_configuration(name, mask, stats_dict, distances_from,
                               softmax_output, y_target, y_output, bias_target, nb_classes)


def infer_predicted_bias(
    biased_base_model: VCBABaseModel,
    nb_prep_epochs: int = 1,
    metric: str = "voronoi"
):
    """
    Trains the vanilla model biased model while computing the distances between:
        - the misclassified samples
        - the Voronoi Hyperplane that separates them from their target class
    The distance computation starts after nb_prep_epochs epochs.
    The statististics of these distances are saved in the file args.epoch_stats_file
    """
    for epoch in range(1, nb_prep_epochs + 1):
        train_biased_model(
            base_model=biased_base_model,
            epoch=epoch,
            measure_dist=False)
    if metric == "voronoi":
        for epoch in range(nb_prep_epochs + 1, biased_base_model.epochs + 1):
            print(f"Computing centroids for epoch {epoch}")
            get_centroids_from_batch(biased_base_model)
            biased_base_model.reweighter.define_hyperplanes()
            train_biased_model(
                base_model=biased_base_model,
                epoch=epoch,
                metric="voronoi"
            )
            # Save both files at every epoch (overwriting the previous one)
            torch.save(biased_base_model.reweighter.stats_dist,
                       args.epoch_stats_file)
    if metric in ["centroid_distance", "centroid_distance_ratio"]:
        print("in centroid distance")
        for epoch in range(nb_prep_epochs + 1, biased_base_model.epochs + 1):
            get_centroids_from_batch(biased_base_model)
            train_biased_model(base_model=biased_base_model,
                               epoch=epoch,
                               metric=metric)
            # Save both files at every epoch (overwriting the previous one)
            torch.save(biased_base_model.reweighter.stats_dist,
                       args.epoch_stats_file)


def train_biased_model(
    base_model: VCBABaseModel,
    epoch: int,
    measure_dist: bool = True,
    metric: str = "voronoi",
):
    """
    Trains a model contained in a BaseModel for 1 epoch.
    Computes and updates the distances to Voronoi Hyperplanes
    if compute_dist_to_hyperplane is set to True.
    """
    base_model.model.train()
    loss_task_tot = AverageMeter("Loss", ":.4e")
    misclassified = 0
    top1 = AverageMeter("Acc@1", ":6.2f")
    tk0 = tqdm(
        base_model.dataloaders["train"],
        total=int(len(base_model.dataloaders["train"])),
        leave=True,
    )
    for _, (data, target, _, idx) in enumerate(tk0):
        data = data.to(base_model.device)
        target = target.to(base_model.device)
        idx = idx.to(base_model.device)
        output = base_model.model(data)
        output_bottleneck = base_model.bottleneck.output
        if measure_dist:
            base_model.reweighter.update(
                output_model=output,
                output_bottleneck=output_bottleneck,
                target=target,
                indices=idx,
                keep_centro=True)
            base_model.reweighter.update_dist_lists(epoch, metric)
        loss_task = torch.mean(base_model.criterion(output, target))
        loss_task_tot.update(loss_task.item(), data.size(0))
        loss_task.backward()
        base_model.optimizer.step()
        base_model.optimizer.zero_grad()
        acc1 = accuracy(output, target, topk=(1,))
        top1.update(acc1[0], data.size(0))
        if measure_dist:
            mask = torch.logical_not(
                base_model.reweighter.stats_dist[:, epoch - 1, 0].isnan())
            avg_dist_h = torch.sum(torch.nan_to_num(base_model.reweighter.stats_dist[:, epoch - 1, 0], nan=0.0)
                                   ) / torch.sum(mask)
            rel_dist = torch.sum(
                torch.nan_to_num(
                    base_model.reweighter.stats_dist[:, epoch - 1, 2], nan=0.0)
            ) / torch.sum(mask)
            misclassified += (
                (target != torch.argmax(output, dim=1)).nonzero().size(0)
            )
            elts_tot = torch.sum((base_model.reweighter.dist_from_hyperplane[:, 0] != float('nan'))
                                 ).item()
            tk0.set_postfix(loss_task=loss_task_tot.avg,
                            top1=top1.avg.item(),
                            d_avg=avg_dist_h.item(),
                            d_rel=rel_dist.item(),
                            m_total=elts_tot,
                            m_epoch=misclassified)
        else:
            tk0.set_postfix(loss_task=loss_task_tot.avg, top1=top1.avg.item())
    return loss_task_tot.avg, top1.avg.item()


def train_test_epoch(
    base_model: VCBABaseModel, epoch: int, predicted_bias_labels: torch.Tensor
):
    train_results = train_with_information_removal(
        base_model, epoch, predicted_bias_labels
    )
    with torch.no_grad():
        base_model.sched.step()

        test_results = test_with_information_removal(
            base_model,
            base_model.dataloaders["test"],
            predicted_bias_labels=predicted_bias_labels,
        )


def rho_class(biased_model: VCBABaseModel, stats_dist_file_per_epoch: str = None):
    """
    Computes :
        - rho_per_class : rho_t of each target class t, ie the proportion of inferred bias-aligned samples in each class
        - predicted_bias_labels : for each sample, its inferred bias label
        - dist_to_class : for each sample, its relative distance to its Voronoi Hyperplane ('nan' if the sample is bias-aligned)
    """
    if stats_dist_file_per_epoch is not None:
        stats_dist = torch.load(stats_dist_file_per_epoch,
                                map_location=biased_model.device)
        print("Stats_dist loaded")
    else:
        stats_dist = biased_model.reweighter.stats_dist.detach()
    non_nan_dist = torch.logical_not(stats_dist[:, :, 2].isnan())
    mean_max_dist = torch.nan_to_num(torch.sum(torch.nan_to_num(
        stats_dist[:, :, 2], nan=0), dim=0)/torch.sum(non_nan_dist, dim=0), nan=0)
    best_epoch = torch.argmax(mean_max_dist) + 1
    print(best_epoch)
    best_epoch_stats_dist = stats_dist[:, best_epoch - 1, :].squeeze(1)
    predicted_bias_labels = best_epoch_stats_dist[:, 1].to(torch.int64)
    print(predicted_bias_labels)
    dist_to_class = best_epoch_stats_dist[:, 0]
    print(dist_to_class)
    mask_aligned = (dist_to_class.isnan())
    # For aligned samples (and misaligned after best epoch)
    # set labels to target labels and distance to 1
    predicted_bias_labels[mask_aligned] = biased_model.nb_classes
    dist_to_class[mask_aligned] = 1
    predicted_bias_labels_onehot = one_hot(
        predicted_bias_labels, num_classes=biased_model.nb_classes + 1
    )[:, : biased_model.nb_classes]
    target_elts_class = count_elements_per_class(basemodel=biased_model)
    elts_class = torch.sum(predicted_bias_labels_onehot, dim=0)
    rho_per_class = (target_elts_class - elts_class) / target_elts_class
    print(rho_per_class)
    return rho_per_class, predicted_bias_labels, dist_to_class


def train_with_information_removal(
    debiased_basemodel: VCBABaseModel, epoch: int, predicted_bias_target: torch.Tensor
):
    """
    Trains a model with reweighted crossentropy
    (based on the predicted_bias_target)
    and conditional mutual information removal
    """
    debiased_basemodel.model.train()
    loss_task_tot = AverageMeter("Loss", ":.4e")
    loss_private_tot = AverageMeter("Loss", ":.4e")
    MI_tot = AverageMeter("Regu", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    private_top1 = AverageMeter("Acc@1", ":6.2f")
    tk0 = tqdm(
        debiased_basemodel.dataloaders["train"],
        total=int(len(debiased_basemodel.dataloaders["train"])),
        leave=True,
    )
    for _, (data, target, bias_target, indices) in enumerate(tk0):
        data = data.to(debiased_basemodel.device)
        target = target.to(debiased_basemodel.device)
        indices = indices.to(debiased_basemodel.device)
        output = debiased_basemodel.model(data)
        output_private = debiased_basemodel.PH()
        output_bottleneck = debiased_basemodel.bottleneck.output
        if debiased_basemodel.reweighter.supervised:
            private_label = bias_target.to(debiased_basemodel.device)
            loss_task = compute_weighted_loss(
                debiased_basemodel,
                output,
                output_bottleneck,
                target,
                indices=indices,
                bias_target=private_label,
            )
        else:
            private_label = predicted_bias_target[indices]
            loss_task = compute_weighted_loss(
                debiased_basemodel, output, output_bottleneck, target, indices=indices
            )
        loss_private = torch.mean(
            debiased_basemodel.criterion(output_private, private_label)
        )
        # if (private_label != target).nonzero().size(0) > 0:
        #     MI = debiased_basemodel.MI(
        #         debiased_basemodel.PH, private_label, mask=(
        #             private_label != target)
        #     )
        #     MI_tot.update(MI.item(), data.size(0))
        #     loss_task_tot.update(loss_task.item(), data.size(0))
        #     loss_private_tot.update(loss_private.item(), data.size(0))
        #     loss = loss_task + loss_private + debiased_basemodel.gamma * MI
        # else:
        #     loss_task_tot.update(loss_task.item(), data.size(0))
        #     loss_private_tot.update(loss_private.item(), data.size(0))
        #     loss = loss_task + loss_private
        # loss.backward()
        # debiased_basemodel.optimizer.step()
        # debiased_basemodel.optimizer.zero_grad()
        # debiased_basemodel.PH_optimizer.step()
        # debiased_basemodel.PH_optimizer.zero_grad()
        if (private_label != target).nonzero().size(0) > 0:
            MI = debiased_basemodel.MI(
                debiased_basemodel.PH, private_label, mask=(
                    private_label != target)
            )
            MI_tot.update(MI.item(), data.size(0))
            loss_task_tot.update(loss_task.item(), data.size(0))
            loss_private_tot.update(loss_private.item(), data.size(0))
            loss = loss_task + debiased_basemodel.gamma * MI
            loss.backward()
            debiased_basemodel.PH_optimizer.zero_grad()
            loss_private.backward()

        else:
            loss_task_tot.update(loss_task.item(), data.size(0))
            loss_private_tot.update(loss_private.item(), data.size(0))
            loss = loss_task + loss_private
        debiased_basemodel.optimizer.step()
        debiased_basemodel.optimizer.zero_grad()
        debiased_basemodel.PH_optimizer.step()
        debiased_basemodel.PH_optimizer.zero_grad()
        acc1 = accuracy(output, target, topk=(1,))
        acc1_private = accuracy(output_private, private_label, topk=(1,))
        top1.update(acc1[0], data.size(0))
        private_top1.update(acc1_private[0], data.size(0))
        tk0.set_postfix(
            loss_y=loss_task_tot.avg,
            top1_y=top1.avg.item(),
            loss_b=loss_private_tot.avg,
            top1_b=private_top1.avg.item(),
            epoch=epoch,
        )
    return {
        "train_loss": loss_task_tot.avg,
        "train_acc": top1.avg.item(),
        "train_bias_loss": loss_private_tot.avg,
        "train_bias_acc": private_top1.avg.item(),
    }


def test_with_information_removal(
    base_model: VCBABaseModel,
    data_loader: DataLoader,
    predicted_bias_labels: torch.Tensor,
):
    """
    Evaluates a model that uses information removal
    """
    base_model.model.eval()
    loss_task_tot = AverageMeter("Loss", ":.4e")
    loss_private_tot = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    private_top1 = AverageMeter("Acc@1", ":6.2f")
    tk0 = tqdm(data_loader, total=int(len(data_loader)), leave=True)
    for _, (data, target, bias_target, indices) in enumerate(tk0):
        data = data.to(base_model.device)
        target = target.to(base_model.device)
        indices = indices.to(base_model.device)
        output = base_model.model(data)
        output_private = base_model.PH()
        if base_model.reweighter.supervised:
            bias_labels = bias_target.to(base_model.device)
        else:
            bias_labels = predicted_bias_labels[indices].to(base_model.device)
        loss_task = torch.mean(base_model.criterion(output, target))
        loss_task_tot.update(loss_task.item(), data.size(0))
        loss_private = torch.mean(
            base_model.criterion(output_private, bias_labels))
        loss_private_tot.update(loss_private.item(), data.size(0))
        acc1 = accuracy(output, target, topk=(1,))
        acc1_private = accuracy(output_private, bias_labels, topk=(1,))
        top1.update(acc1[0], data.size(0))
        private_top1.update(acc1_private[0], data.size(0))
        tk0.set_postfix(
            loss_task=loss_task_tot.avg,
            loss_bias=loss_private_tot.avg,
            top1=top1.avg.item(),
            top1_bias=private_top1.avg.item(),
        )
        tk0.update()

    return {
        "train_loss": loss_task_tot.avg,
        "train_acc": top1.avg.item(),
        "train_bias_loss": loss_private_tot.avg,
        "train_bias_acc": private_top1.avg.item(),
    }


def count_elements_per_class(basemodel: VCBABaseModel):
    """
    Counts the number of samples of each target class in the dataset
    """
    tk0 = tqdm(basemodel.dataloaders["train"],
               total=int(len(basemodel.dataloaders["train"])),
               leave=True,)
    elements_per_class = torch.zeros(size=(1, basemodel.nb_classes),
                                     device=basemodel.device,
                                     requires_grad=False)
    for _, (_, target, _, _) in enumerate(tk0):
        target = target.to(basemodel.device)
        target_one_hot = one_hot(target, num_classes=basemodel.nb_classes)
        elements_per_class += torch.sum(target_one_hot, dim=0)
    return elements_per_class


def build_predicted_bias_labels(
    predicted_bias_labels: torch.Tensor,
    train_loader: DataLoader,
    device: torch.device,
    num_classes: int = 2,
):
    """
    Builds the tensor containing the inferred bias labels
    """
    tk0 = tqdm(train_loader, total=int(len(train_loader)), leave=True)
    predicted_bias_target = torch.zeros_like(predicted_bias_labels)
    for _, (_, target, bias_target, indices) in enumerate(tk0):
        target = target.to(device)
        bias_target = bias_target.to(device)
        indices = indices.to(device)
        predicted_bias_target[indices] = predicted_bias_labels[indices] * (
            predicted_bias_labels[indices] != num_classes
        ) + target * (predicted_bias_labels[indices] == num_classes)
    return predicted_bias_target
