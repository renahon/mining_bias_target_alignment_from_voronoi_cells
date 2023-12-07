# File:                     reweighting.py
# Created by:               Rémi Nahon
# Last revised by:          Rémi Nahon
#            date:          2022/3/15
#
# Reweighter class used to measure distance to Voronoi Hyperplanes during
# training and to compute the weights used in the reweighted Crossentropy
# ================================= IMPORTS =================================
import torch
from torch.nn.functional import one_hot
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import square, sqrt, sum
from utils.tools import print_remi, add_dims_index
# ================================== CODE ===================================


class Centroids(object):
    def __init__(
        self,
        nb_features: int,
        nb_classes: int,
        device,
        represented_samples: str = "correct",
    ) -> None:
        self.device = device
        self.represented_samples = represented_samples
        self.nb_features = nb_features
        self.nb_classes = nb_classes
        self.nb_elements = torch.zeros(
            size=(1, self.nb_classes),
            device=self.device,
            requires_grad=False,
            dtype=torch.int32,
        )
        self.positions = torch.zeros(
            (self.nb_features, self.nb_classes), device=self.device, requires_grad=False
        )
        self.dist_to_zero = torch.zeros(
            size=(1, self.nb_classes), device=self.device, requires_grad=False
        )
        self.all_updates_complete = True
        pass

    def _reset(self):
        self.all_updates_complete = True
        pass

    def update(self, output_bottleneck, y_target=None, y_output=None, bias_target=None):
        if self.represented_samples == "correct":
            # update from the correctly classified samples
            samples_one_hot = torch.logical_and(
                one_hot(y_output, self.nb_classes), one_hot(
                    y_target, self.nb_classes)
            )
        elif self.represented_samples == "output":
            samples_one_hot = one_hot(y_output, self.nb_classes)
        elif self.represented_samples == "bias":
            samples_one_hot = one_hot(bias_target, self.nb_classes)

        if torch.prod(self.nb_elements + torch.sum(samples_one_hot, dim=0)) > 1:
            self.positions = (
                self.nb_elements * self.positions
                + torch.sum(
                    output_bottleneck.unsqueeze(-1) *
                    samples_one_hot.unsqueeze(1),
                    dim=0,
                )
            ) / (self.nb_elements + torch.sum(samples_one_hot, dim=0))

            self.nb_elements += torch.sum(samples_one_hot, dim=0)
        else:
            self.all_updates_complete = False

    def dist_from_samples(
        self,
        y_target,
        y_output,
        output_bottleneck,
        bias_target=None,
        dist_from="target_class",
        samples_type="misclassified"
    ):
        if samples_type == "misclassified":
            samples_mask = (y_target != y_output)
        elif samples_type == "all":
            samples_mask = (y_target == y_target)
        if dist_from == "inference":
            out_one_hot = add_dims_index(samples_mask, nb_dims=2, index=-1) * \
                one_hot(y_output, self.nb_classes).unsqueeze(-1)
            samples_one_hot = out_one_hot
        if dist_from == "target_class":
            target_one_hot = add_dims_index(samples_mask, nb_dims=2, index=-1) * \
                one_hot(y_target, self.nb_classes).unsqueeze(-1)
            samples_one_hot = target_one_hot
        if dist_from == "bias":
            target_one_hot = add_dims_index(samples_mask, nb_dims=2, index=-1) * \
                one_hot(bias_target, self.nb_classes).unsqueeze(-1)
            samples_one_hot = target_one_hot
        out_representation = output_bottleneck.unsqueeze(1)
        centros = torch.transpose(self.positions, 0, 1).unsqueeze(0)
        dist = torch.full(size=y_target.size(),
                          dtype=torch.float,
                          fill_value=float('nan'),
                          device=self.device)

        if torch.sum(samples_mask) >= 1:
            dist[samples_mask] = sqrt(sum(sum(square(
                samples_one_hot[samples_mask, :, :] *
                out_representation[samples_mask, :, :] - centros), 1), 1))
        return dist


class VCBAReweighter(object):
    def __init__(
        self,
        nb_features: int,
        target_classes: torch.Tensor,
        batch_size: int,
        epochs: int,
        dataset_size: int,
        device="cuda:0",
        bias_inference_metric="voronoi",
    ):
        self.nb_features = nb_features
        self.device = device
        self.target_classes = target_classes
        self.nb_classes = self.target_classes.size(0)
        self.nb_epochs = epochs
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.bias_inference_metric = bias_inference_metric
        self.supervised = False
        self.reset()
        self.reset_hyperplane_dist()

    def reset(self):
        self.correct_centroids = Centroids(
            nb_features=self.nb_features,
            nb_classes=self.nb_classes,
            device=self.device,
            represented_samples="correct",
        )
        if self.bias_inference_metric in ["centroid_distance", "centroid_distance_ratio", "stats"]:
            self.inference_centroids = Centroids(
                nb_features=self.nb_features,
                nb_classes=self.nb_classes,
                device=self.device,
                represented_samples="output",
            )
        if self.bias_inference_metric == "stats":
            self.bias_centroids = Centroids(
                nb_features=self.nb_features,
                nb_classes=self.nb_classes,
                device=self.device,
                represented_samples="bias")
        self.do_weighing = True
        self.reset_weights()

    def reset_weights(self):
        self.weights = torch.ones(
            self.batch_size, device=self.device, requires_grad=False
        )

    def reset_hyperplane_dist(self):
        self.dist_from_hyperplane = torch.full(
            size=(self.dataset_size, 7),
            fill_value=float('nan'),
            device=self.device,
            requires_grad=False)

        self.stats_dist = torch.full(
            size=(self.dataset_size, self.nb_epochs, 3),
            fill_value=float('nan'),
            dtype=torch.float,
            requires_grad=False,
            device=self.device,
        )

    def reset_centroids_dist(
            self
    ):
        # distance from between misclassified samples and :
        # - row 1 : the correct centroid of their home class
        # - row 2 : the model centroid of their "bias class"
        # - row 3 : the centroids and the center (coordinates (0,0,...,0,0))
        self.dist_from_centroids = torch.full(size=(self.dataset_size, 2),
                                              fill_value=float('nan'),
                                              device=self.device,
                                              requires_grad=False)

    def set_predicted_bias_labels(
            self, predicted_bias_target: torch.Tensor, predicted_rho: torch.Tensor
    ):
        self.predicted_bias_target = predicted_bias_target.detach()
        self.predicted_rho = predicted_rho.detach()

    def set_supervised(self, predicted_rho: torch.Tensor):
        self.predicted_rho = predicted_rho.detach()
        self.supervised = True

    def update(
        self,
        output_model: torch.Tensor,
        output_bottleneck: torch.Tensor,
        target: torch.Tensor,
        bias_target=None,
        indices=None,
        keep_centro=False,
    ):
        """Updates the Reweighter with the new outputs of the model for the minibatch"""
        self.output_model = output_model.detach()
        self.y_target = target.detach()
        self.output_bottleneck = (
            output_bottleneck.clone().detach().reshape((-1, self.nb_features))
        )
        self.y_output = torch.argmax(self.output_model, dim=1)
        if bias_target is not None:
            self.bias_target = bias_target.detach()
        if indices is not None:
            self.indices = indices.detach()
        if not keep_centro:
            self.correct_centroids.update(
                output_bottleneck=self.output_bottleneck,
                y_target=self.y_target,
                y_output=self.y_output)
            if self.bias_inference_metric in ["centroid_distance",
                                              "centroid_distance_ratio",
                                              "stats"]:
                self.inference_centroids.update(
                    output_bottleneck=self.output_bottleneck,
                    y_target=self.y_target,
                    y_output=self.y_output)
            if self.bias_inference_metric == "stats":
                # print(f"bias device = {bias_target.get_device()}")
                self.bias_centroids.update(
                    output_bottleneck=self.output_bottleneck,
                    bias_target=self.bias_target)

    def define_hyperplanes(self):
        """Computes the Voronoi Hyperplanes that separate the different classes"""
        self.w_h = torch.zeros(
            size=(self.nb_classes, self.nb_classes, self.nb_features),
            device=self.device,
            requires_grad=False,
        )
        self.b_h = torch.zeros(
            size=(self.nb_classes, self.nb_classes),
            device=self.device,
            requires_grad=False,
        )
        for i in range(self.nb_classes):
            for j in range(self.nb_classes):
                if i != j:
                    self.w_h[i, j, :] = (
                        self.correct_centroids.positions[:, j]
                        - self.correct_centroids.positions[:, i]
                    )
                    self.w_h[j, i, :] = self.w_h[i, j, :]
                    self.b_h[i, j] = 0.5 * torch.sum(
                        (
                            torch.pow(
                                self.correct_centroids.positions[:, i], 2)
                            - torch.pow(self.correct_centroids.positions[:, j], 2)
                        )
                    )
                    self.b_h[j, i] = self.b_h[i, j]

    def compute_dist_from_centroids(self):
        """
        Computes and logs the distances of each misclassified sample
        to their target class centroid and to the centroid of the class
        they are assigned to.
        """
        if self.bias_inference_metric == "stats":
            samples_to_evaluate = "all"
        else:
            samples_to_evaluate = "misclassified"
        dist_from_correct_centroids = self.correct_centroids.dist_from_samples(
            y_target=self.y_target,
            y_output=self.y_output,
            output_bottleneck=self.output_bottleneck,
            dist_from="target_class",
            samples_type=samples_to_evaluate)
        dist_from_inferred_centroids = self.inference_centroids.dist_from_samples(
            y_target=self.y_target,
            y_output=self.y_output,
            output_bottleneck=self.output_bottleneck,
            dist_from="inference",
            samples_type=samples_to_evaluate)

        dist_centroids_center = torch.norm(
            self.correct_centroids.positions,
            p=2,
            dim=0)
        if self.bias_inference_metric == "stats":
            dist_from_bias_centroids = self.bias_centroids.dist_from_samples(
                y_target=self.y_target,
                y_output=self.y_output,
                bias_target=self.bias_target,
                output_bottleneck=self.output_bottleneck,
                dist_from="bias",
                samples_type="all")
            return dist_from_correct_centroids, dist_from_inferred_centroids, dist_centroids_center, dist_from_bias_centroids
        return dist_from_correct_centroids, dist_from_inferred_centroids, dist_centroids_center

    def update_dist_lists(self, e, metric_type):
        if metric_type == "voronoi":
            dist, rel_dist = self.compute_dist_from_hyperplane()
            self.stats_dist[self.indices, e - 1, 0] = dist
            self.stats_dist[self.indices, e - 1,
                            1] = self.y_output.to(torch.float)
            self.stats_dist[self.indices, e - 1, 2] = rel_dist
        elif metric_type == "centroid_distance":
            dist_correct, _, dist_centroids_center = self.compute_dist_from_centroids()
            rel_dist = dist_correct/torch.mean(dist_centroids_center)
            self.stats_dist[self.indices, e - 1, 0] = dist_correct
            self.stats_dist[self.indices, e - 1,
                            1] = self.y_output.to(torch.float)
            self.stats_dist[self.indices, e - 1, 2] = rel_dist
        elif metric_type == "centroid_distance_ratio":
            dist_correct, dist_inference, dist_centroids_center = self.compute_dist_from_centroids()
            dist_ratio = dist_correct/dist_inference
            rel_dist = dist_ratio
            self.stats_dist[self.indices, e - 1, 0] = dist_ratio
            self.stats_dist[self.indices, e - 1,
                            1] = self.y_output.to(torch.float)
            self.stats_dist[self.indices, e - 1, 2] = rel_dist
        elif metric_type == "stats":
            dist_correct_centroids, dist_inference_centroids, dist_centroids_center, dist_bias_centroids = self.compute_dist_from_centroids()
            dist_hyperplane, _ = self.compute_dist_from_hyperplane()
            return {'correct_centroids': dist_correct_centroids,
                    'inference_centroids': dist_inference_centroids,
                    'bias_centroids': dist_bias_centroids,
                    'centroids_center': dist_centroids_center,
                    'hyperplane': dist_hyperplane, }

    def compute_dist_from_hyperplane(self):
        misaligned = self.y_target != self.y_output
        w = self.w_h[self.y_target, self.y_output]
        norm_w = torch.norm(w, p=2, dim=1)  
        b = self.b_h[self.y_target, self.y_output]
        dist_centroids_center = torch.norm(self.correct_centroids.positions,
                                           p=2,
                                           dim=0)
        dist = torch.full_like(b,
                               fill_value=float('nan'),
                               device=self.device)
        if self.bias_inference_metric != "stats":
            mask = (norm_w != 0)*misaligned
        else:
            mask = norm_w != 0
        dist[mask] = torch.abs(torch.einsum(
            'ij,ij->i', w, self.output_bottleneck)+b)[mask]/norm_w[mask]
        rel_dist = torch.full_like(
            dist, fill_value=float('nan'), device=self.device)
        rel_dist[mask] = dist[mask] * self.nb_classes / \
            torch.sum(dist_centroids_center)
        # print(dist.type())
        return dist, rel_dist

    def update_dist_from_hyperplane(self, epoch):
        """
        Computes and logs the distances of each misclassified sample
        to the Voronoi Hyperplane that separates it from its target class
        """
        for i in range(self.y_target.size(0)):
            y_t = self.y_target[i]
            y_o = self.y_output[i]
            idx_i = self.indices[i]
            if y_t != y_o:
                self.misaligned_at_epoch[idx_i, epoch - 1] = 1
                out_bot = self.output_bottleneck[i, :]
                w = self.w_h[y_t, y_o]
                norm_w = torch.norm(w, p=2)
                if norm_w > 0:
                    b = self.b_h[y_t, y_o]
                    dist = torch.abs(torch.matmul(w, out_bot) + b) / norm_w
                    if self.dist_from_hyperplane[idx_i, 0] == 'nan':
                        self.dist_from_hyperplane[idx_i, 2] = epoch
                    self.stats_dist[idx_i, epoch - 1, 0] = dist
                    self.stats_dist[idx_i, epoch - 1, 1] = y_o
                    self.stats_dist[idx_i, epoch - 1, 2] = (dist
                                                            * self.nb_classes
                                                            / torch.sum(
                                                                torch.norm(
                                                                    self.correct_centroids.positions, p=2, dim=0)
                                                            )
                                                            )
                    self.centro_dist[:, epoch - 1] = torch.norm(
                        self.correct_centroids.positions, p=2, dim=0
                    )
                    self.dist_from_hyperplane[idx_i, 0] = dist
                    self.dist_from_hyperplane[idx_i, 1] = y_o
                    self.dist_from_hyperplane[idx_i, 3] = self.nb_classes / torch.sum(
                        torch.norm(
                            self.correct_centroids.positions, p=2, dim=1)
                    )

    def _compute_weights(self, bias_target: torch.Tensor = None):
        """
        Computes the weights used in the reweighted crossentropy,
        in such a way that for x_i, of index i and target class = c :
        - if x_i is bias-misaligned , self.weights[i] = 1/(1-rho_c)
        - else, self.weights[i] = 1/rho_c
        """
        rho_by_class = self.predicted_rho
        if torch.prod(rho_by_class) * torch.prod(1 - rho_by_class) > 0:
            # if each rho_c is neither 0 nor 1, some but not all samples of each class are bias misaligned
            if bias_target is not None:
                pbt_minibatch = bias_target
                aligned_samples = (
                    pbt_minibatch == self.y_target).type(torch.uint8)
            else:
                pbt_minibatch = self.predicted_bias_target[self.indices]
                aligned_samples = (
                    pbt_minibatch == self.nb_classes).type(torch.uint8)
            score_aligned = (1 / rho_by_class).squeeze(0)
            score_misaligned = (1 / (1 - rho_by_class)).squeeze(0)
            misaligned_samples = 1 - aligned_samples
            pbt_aligned_oh = aligned_samples.unsqueeze(1) * one_hot(
                self.y_target, num_classes=self.nb_classes
            )
            pbt_misaligned_oh = (
                misaligned_samples.unsqueeze(1)
                * one_hot(pbt_minibatch, num_classes=self.nb_classes + 1)[
                    :, : self.nb_classes
                ]
            )
            # Computation of the weights
            self.weights = torch.sum(
                score_aligned * pbt_aligned_oh + score_misaligned * pbt_misaligned_oh,
                dim=1,
            )

    def weigh(self, reweigh: bool = True, bias_target: torch.Tensor = None):
        self.reset_weights()
        if self.do_weighing and reweigh:
            self._compute_weights(bias_target=bias_target)
        return self.weights

    def get_rho_from_bias_labels(self, dataloader: DataLoader):
        tk0 = tqdm(dataloader, total=int(len(dataloader)), leave=True)
        class_counter = torch.zeros(
            size=(1, self.nb_classes), device=self.device, requires_grad=False
        )
        bias_aligned_counter = torch.zeros(
            size=(1, self.nb_classes), device=self.device, requires_grad=False
        )
        for _, (data, target, bias_target, _) in enumerate(tk0):
            data = data.to(self.device)
            target = target.to(self.device)
            bias_target = bias_target.to(self.device)
            target_oh = one_hot(target)
            bias_target_oh = one_hot(bias_target)
            class_counter += torch.sum(target_oh, dim=0)
            bias_aligned_counter += torch.sum(target_oh *
                                              bias_target_oh, dim=0)
            tk0.set_postfix(
                nb_0=class_counter[0, 0].item(),
                bias_0=bias_aligned_counter[0, 0].item(),
            )
        return bias_aligned_counter / class_counter
