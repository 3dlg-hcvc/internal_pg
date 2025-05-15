import os

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from minsu3d.common_ops.functions import common_ops
from minsu3d.evaluation.instance_segmentation import GeneralDatasetEvaluator
from minsu3d.evaluation.object_detection import evaluate_bbox_acc
from minsu3d.loss.pt_offset_loss import PTOffsetLoss
from minsu3d.model.module import Backbone, BackboneFPN
from minsu3d.util.io import save_prediction
from minsu3d.util.lr_decay import cosine_lr_decay
from torch.nn import functional as F

import MinkowskiEngine as ME


class GeneralModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        input_channel = 3 + cfg.model.network.use_color * 3 + cfg.model.network.use_normal * 3
        if cfg.model.network.use_fpn:
            self.backbone = BackboneFPN(
                input_channel=input_channel, output_channel=cfg.model.network.m, block_channels=cfg.model.network.blocks,
                block_reps=cfg.model.network.block_reps, sem_classes=cfg.data.classes, use_gamma=cfg.model.network.use_gamma
            )
        else:
            self.backbone = Backbone(
                input_channel=input_channel, output_channel=cfg.model.network.m, block_channels=cfg.model.network.blocks,
                block_reps=cfg.model.network.block_reps, sem_classes=cfg.data.classes
            )
        self.val_test_step_outputs = []
        self.cfg = cfg

    def configure_optimizers(self):
        if hasattr(self, 'existing_param_keys') and hasattr(self, 'missing_param_keys'):
            new_params = []
            existing_params = []

            for name, param in self.named_parameters():
                param_key = name

                if param_key in self.existing_param_keys:
                    existing_params.append(param)
                    print(f"Parameter with lower LR: {name}")
                else:
                    new_params.append(param)
                    print(f"Parameter with higher LR: {name}")

            param_groups = []

            if new_params:
                param_groups.append({
                    'params': new_params, 
                    'lr': getattr(self, 'new_params_lr', self.cfg.model.optimizer.lr)
                })

            if existing_params:
                param_groups.append({
                    'params': existing_params, 
                    'lr': getattr(self, 'existing_params_lr', self.cfg.model.optimizer.lr * 0.1)
                })

            if self.cfg.model.optimizer.name == 'Adam':
                optimizer = torch.optim.Adam(
                    param_groups,
                )
            elif self.cfg.model.optimizer.name == 'AdamW':
                optimizer = torch.optim.AdamW(
                    param_groups,
                )
            else:
                raise NotImplementedError
        else:
            optimizer = hydra.utils.instantiate(self.cfg.model.optimizer, params=self.parameters())

        return optimizer

    def forward(self, data_dict):
        input_dict = {"coord": data_dict["point_xyz"]}
        input_dict["feat"] = data_dict["point_xyz"]
        if self.cfg.model.network.use_color:
            input_dict["feat"] = torch.cat((input_dict["feat"], data_dict["point_color"]), dim=2)
        if self.cfg.model.network.use_normal:
            input_dict["feat"] = torch.cat((input_dict["feat"], data_dict["point_normal"]), dim=2)
        backbone_output_dict = self.backbone(input_dict)
        return backbone_output_dict

    def _loss(self, data_dict, output_dict):
        losses = {}
        """semantic loss"""
        losses["semantic_loss"] = torch.nn.functional.cross_entropy(
            output_dict["semantic_scores"], data_dict["sem_labels"].long(), ignore_index=-1
        )

        """offset loss"""

        gt_offsets = data_dict["instance_center_xyz"] - torch.flatten(data_dict["point_xyz"], end_dim=1)
        valid = data_dict["instance_ids"] != -1
        pt_offset_criterion = PTOffsetLoss()
        losses["offset_norm_loss"], losses["offset_dir_loss"] = pt_offset_criterion(
            output_dict["point_offsets"], gt_offsets, valid_mask=valid
        )

        if self.cfg.model.network.use_gamma:
            # Gamma losses
            gt_motion_types = data_dict["instance_motion_types"].reshape(-1)

            if self.hparams.cfg.model.network.use_projection_origin:
                gt_axis_offsets = data_dict["instance_origin_offsets"].reshape(-1, 3)
            else:
                gt_axis_offsets = data_dict["instance_axis_offsets"].reshape(-1, 3)
            gt_directions = data_dict["instance_axis_directions"].reshape(-1, 3)
            valid = torch.logical_and(data_dict["instance_ids"] != -1, gt_motion_types != 2)

            gamma_offset_norm_loss, gamma_offset_dir_loss = pt_offset_criterion(
                output_dict["gamma_offsets"], gt_axis_offsets, valid_mask=valid
            )

            losses["gamma_offset_norm_loss"] = self.cfg.model.network.motion_losses_weight * gamma_offset_norm_loss
            losses["gamma_offset_dir_loss"] = self.cfg.model.network.motion_losses_weight * gamma_offset_dir_loss

            gamma_direction_norm_loss, gamma_direction_dir_loss = pt_offset_criterion(
                output_dict["gamma_directions"], gt_directions, valid_mask=valid
            )

            losses["gamma_direction_norm_loss"] = self.cfg.model.network.motion_losses_weight * gamma_direction_norm_loss
            losses["gamma_direction_dir_loss"] = self.cfg.model.network.motion_losses_weight * gamma_direction_dir_loss

            if self.cfg.model.network.use_gamma_ce:
                gamma_motion_type_loss = torch.nn.functional.cross_entropy(
                    output_dict["gamma_motion_scores"], gt_motion_types.long(), ignore_index=2
                )

                losses["gamma_motion_type_loss"] = self.cfg.model.network.motion_losses_weight * gamma_motion_type_loss

            else:
                # From https://github.com/qiaojunyu/GAMMA-ICRA2024/blob/master/visual_model/losses.py
                def focal_loss(
                    inputs: torch.Tensor,
                    targets: torch.Tensor,
                    alpha: torch.Tensor = None,
                    gamma: float = 2.0,
                    reduction: str = "mean",
                    ignore_index: int = -100,
                ) -> torch.Tensor:
                    if ignore_index is not None:
                        valid_mask = targets != ignore_index
                        targets = targets[valid_mask]

                        if targets.shape[0] == 0:
                            return torch.tensor(0.0).to(dtype=inputs.dtype, device=inputs.device)

                        inputs = inputs[valid_mask]

                    log_p = F.log_softmax(inputs, dim=-1)
                    ce_loss = F.nll_loss(
                        log_p, targets.long(), weight=alpha, ignore_index=ignore_index, reduction="none"
                    )

                    log_p_t = log_p.gather(1, targets[:, None].long()).squeeze(-1)
                    loss = ce_loss * ((1 - log_p_t.exp()) ** gamma)

                    if reduction == "mean":
                        loss = loss.mean()
                    elif reduction == "sum":
                        loss = loss.sum()

                    return loss

                def dice_loss(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
                    input_soft = F.softmax(input, dim=1)

                    target_one_hot = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

                    intersection = torch.sum(input_soft * target_one_hot, dim=0)

                    cardinality = torch.sum(input_soft, dim=0) + torch.sum(target_one_hot, dim=0)

                    dice_score = 2.0 * intersection / (cardinality + eps)

                    return torch.mean(-dice_score + 1.0)

                def one_hot(
                    labels: torch.Tensor,
                    num_classes: int,
                    device: torch.device = None,
                    dtype: torch.dtype = None,
                    eps: float = 1e-6,
                ) -> torch.Tensor:
                    if not isinstance(labels, torch.Tensor):
                        raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

                    labels_64 = labels.clone().to(torch.int64)

                    if num_classes < 1:
                        raise ValueError("The number of classes must be bigger than one." f" Got: {num_classes}")

                    shape = labels_64.shape
                    if device is None:
                        device = labels_64.device

                    one_hot_tensor = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

                    safe_labels = torch.clamp(labels_64, 0, num_classes - 1)

                    result = one_hot_tensor.scatter_(1, safe_labels.unsqueeze(1), 1.0) + eps

                    return result

                focal = focal_loss(output_dict["gamma_motion_scores"], gt_motion_types, gamma=2.0, alpha=None, reduction="mean", ignore_index=2)

                dice = dice_loss(output_dict["gamma_motion_scores"], gt_motion_types)

                losses["gamma_motion_type_loss"] = self.cfg.model.network.motion_losses_weight * (focal + dice)

        return losses

    def training_step(self, data_dict, idx):
        output_dict = self(data_dict)
        losses = self._loss(data_dict, output_dict)
        total_loss = 0
        for loss_name, loss_value in losses.items():
            total_loss += loss_value
            self.log(
                f"train/{loss_name}", loss_value, on_step=False, sync_dist=True,
                on_epoch=True, batch_size=len(data_dict["scan_ids"])
            )
        self.log(
            "train/total_loss", total_loss, on_step=False, sync_dist=True,
            on_epoch=True, batch_size=len(data_dict["scan_ids"])
        )
        return total_loss

    def on_train_epoch_end(self):
        cosine_lr_decay(
            self.trainer.optimizers[0], self.hparams.cfg.model.optimizer.lr, self.current_epoch,
            self.hparams.cfg.model.lr_decay.decay_start_epoch, self.hparams.cfg.model.trainer.max_epochs, 1e-6
        )

    def validation_step(self, data_dict, idx):
        pass

    def on_validation_epoch_end(self):
        # evaluate instance predictions
        if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
            all_pred_insts = []
            all_gt_insts = []
            all_gt_insts_bbox = []
            for pred_instances, gt_instances, gt_instances_bbox in self.val_test_step_outputs:
                all_gt_insts_bbox.append(gt_instances_bbox)
                all_pred_insts.append(pred_instances)
                all_gt_insts.append(gt_instances)
            self.val_test_step_outputs.clear()
            inst_seg_evaluator = GeneralDatasetEvaluator(
                self.hparams.cfg.data.class_names, -1, self.hparams.cfg.data.ignore_classes
            )
            inst_seg_eval_result = inst_seg_evaluator.evaluate(all_pred_insts, all_gt_insts, print_result=False)

            obj_detect_eval_result = evaluate_bbox_acc(
                all_pred_insts, all_gt_insts_bbox, self.hparams.cfg.data.class_names,
                self.hparams.cfg.data.ignore_classes, print_result=False
            )

            self.log("val_eval/AP", inst_seg_eval_result["all_ap"], sync_dist=True)
            self.log("val_eval/AP 50%", inst_seg_eval_result['all_ap_50%'], sync_dist=True)
            self.log("val_eval/AP 25%", inst_seg_eval_result["all_ap_25%"], sync_dist=True)
            self.log("val_eval/BBox AP 25%", obj_detect_eval_result["all_bbox_ap_0.25"]["avg"], sync_dist=True)
            self.log("val_eval/BBox AP 50%", obj_detect_eval_result["all_bbox_ap_0.5"]["avg"], sync_dist=True)

    def test_step(self, data_dict, idx):
        pass

    def on_test_epoch_end(self):
        # evaluate instance predictions
        if self.current_epoch > self.hparams.cfg.model.network.prepare_epochs:
            all_pred_insts = []
            all_gt_insts = []
            all_gt_insts_bbox = []
            all_sem_acc = []
            all_sem_miou = []
            for semantic_accuracy, semantic_mean_iou, pred_instances, gt_instances, gt_instances_bbox in self.val_test_step_outputs:
                all_sem_acc.append(semantic_accuracy)
                all_sem_miou.append(semantic_mean_iou)
                all_gt_insts_bbox.append(gt_instances_bbox)
                all_gt_insts.append(gt_instances)
                all_pred_insts.append(pred_instances)

            if self.hparams.cfg.model.inference.evaluate:
                inst_seg_evaluator = GeneralDatasetEvaluator(
                    self.hparams.cfg.data.class_names, -1, self.hparams.cfg.data.ignore_classes
                )
                self.print("Evaluating instance segmentation ...")
                inst_seg_eval_result = inst_seg_evaluator.evaluate(all_pred_insts, all_gt_insts, print_result=True)
                obj_detect_eval_result = evaluate_bbox_acc(
                    all_pred_insts, all_gt_insts_bbox,
                    self.hparams.cfg.data.class_names, self.hparams.cfg.data.ignore_classes, print_result=True
                )

                sem_miou_avg = np.mean(np.array(all_sem_miou))
                sem_acc_avg = np.mean(np.array(all_sem_acc))
                self.print(f"Semantic Accuracy: {sem_acc_avg}")
                self.print(f"Semantic mean IoU: {sem_miou_avg}")

                self.val_test_step_outputs.clear()

            if self.hparams.cfg.model.inference.save_predictions:
                save_dir = os.path.join(
                    self.hparams.cfg.exp_output_root_path, 'inference', self.hparams.cfg.model.inference.split,
                    'predictions'
                )
                save_prediction(
                    save_dir, all_pred_insts, self.hparams.cfg.data.mapping_classes_ids,
                    self.hparams.cfg.data.ignore_classes
                )
                self.print(f"\nPredictions saved at {os.path.abspath(save_dir)}")


def clusters_voxelization(clusters_idx, clusters_offset, feats, coords, scale, spatial_shape, device):

    batch_idx = clusters_idx[:, 0]
    c_idxs = clusters_idx[:, 1]
    feats = feats[c_idxs]
    clusters_coords = coords[c_idxs]

    clusters_coords_mean = common_ops.sec_mean(clusters_coords, clusters_offset)  # (nCluster, 3)
    clusters_coords_mean_all = torch.index_select(clusters_coords_mean, 0, batch_idx)  # (sumNPoint, 3)
    clusters_coords -= clusters_coords_mean_all

    clusters_coords_min = common_ops.sec_min(clusters_coords, clusters_offset)
    clusters_coords_max = common_ops.sec_max(clusters_coords, clusters_offset)

    # 0.01 to ensure voxel_coords < spatial_shape
    clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / spatial_shape).max(1)[0] - 0.01
    clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

    min_xyz = clusters_coords_min * clusters_scale[:, None]
    max_xyz = clusters_coords_max * clusters_scale[:, None]

    clusters_scale = torch.index_select(clusters_scale, 0, batch_idx)

    clusters_coords = clusters_coords * clusters_scale[:, None]

    range = max_xyz - min_xyz
    offset = -min_xyz + torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3, device=device)
    offset += torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3, device=device)
    offset = torch.index_select(offset, 0, batch_idx)
    clusters_coords += offset

    clusters_coords = clusters_coords.int()

    batched_xyz = torch.cat((clusters_idx[:, 0].unsqueeze(-1), clusters_coords), dim=1)

    voxel_xyz, voxel_features, _, voxel_point_map = ME.utils.sparse_quantize(
        batched_xyz.float(), feats, return_index=True, return_inverse=True, device=device.type
    )

    clusters_voxel_feats = ME.SparseTensor(features=voxel_features, coordinates=voxel_xyz, device=device)

    return clusters_voxel_feats, voxel_point_map


def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
    """
    Args:
        scores: (N), float, 0~1

    Returns:
        segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
    """
    fg_mask = scores > fg_thresh
    bg_mask = scores < bg_thresh
    interval_mask = (fg_mask == 0) & (bg_mask == 0)

    segmented_scores = (fg_mask > 0).float()
    k = 1 / (fg_thresh - bg_thresh)
    b = bg_thresh / (bg_thresh - fg_thresh)
    segmented_scores[interval_mask] = scores[interval_mask] * k + b

    return segmented_scores
