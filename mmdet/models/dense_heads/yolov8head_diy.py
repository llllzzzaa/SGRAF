# # Copyright (c) OpenMMLab. All rights reserved.
# import math
# import copy
# from typing import List, Optional, Sequence, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor

# from mmengine.config import ConfigDict
# from mmengine.model import bias_init_with_prob
# from mmengine.structures import InstanceData

# from mmdet.registry import MODELS, TASK_UTILS
# from mmdet.models.utils import multi_apply
# from mmdet.utils import reduce_mean
# from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
# from ..task_modules.prior_generators import MlvlPointGenerator
# from ..task_modules.samplers import PseudoSampler
# from ..utils import multi_apply as mmd_multi_apply
# from .base_dense_head import BaseDenseHead


# def distribution_focal_loss(pred_logits: Tensor,
#                             target: Tensor,
#                             reg_max: int) -> Tensor:
#     """
#     Compute Distribution Focal Loss (DFL) between pred_logits and target float distances.

#     pred_logits: (N, 4, reg_max) logits for each side
#     target: (N, 4) float distances (non-negative)
#     """
#     # pred_logits: [N, 4, reg_max]
#     # target: [N, 4]
#     n, _, m = pred_logits.shape  # m == reg_max
#     device = pred_logits.device
#     # clamp targets to [0, reg_max - 1 - eps]
#     t = target.clamp(min=0.0, max=float(m - 1 - 1e-6))
#     lower = t.floor().long()  # [N,4]
#     upper = lower + 1
#     upper_weight = t - lower.float()  # alpha
#     lower_weight = 1.0 - upper_weight

#     # log softmax over last dim
#     logp = F.log_softmax(pred_logits, dim=-1)  # [N,4,m]
#     # gather
#     # build index tensors
#     lower = lower.unsqueeze(-1)  # [N,4,1]
#     upper = upper.unsqueeze(-1)
#     lower_logp = torch.gather(logp, dim=-1, index=lower).squeeze(-1)  # [N,4]
#     upper_logp = torch.gather(logp, dim=-1, index=upper).squeeze(-1)  # [N,4]
#     loss = - (lower_weight * lower_logp + upper_weight * upper_logp)  # [N,4]
#     return loss.sum()  # sum over N and 4 sides


# def dist_to_bbox(dist: Tensor, priors: Tensor) -> Tensor:
#     """
#     Convert distribution distances (expected values per side) to bbox xyxy via priors.
#     dist: [N, 4] expected distances (l, t, r, b) relative to stride (or absolute depending on priors)
#     priors: [N, 4] priors format (cx, cy, w, h) where w/h are stride-scaled cell sizes
#     Return: [N,4] xyxy
#     """
#     # assuming dist in same scale as priors[..., 2:4] (width/height)
#     cx = priors[:, 0]
#     cy = priors[:, 1]
#     w = priors[:, 2]
#     h = priors[:, 3]
#     l = dist[:, 0]
#     t = dist[:, 1]
#     r = dist[:, 2]
#     b = dist[:, 3]
#     x1 = cx - l
#     y1 = cy - t
#     x2 = cx + r
#     y2 = cy + b
#     return torch.stack([x1, y1, x2, y2], dim=-1)


# @MODELS.register_module()
# class YOLOv8Head(BaseDenseHead):
#     """
#     YOLOv8-style head adapted to MMDetection (BaseDenseHead interface).
#     Supports DFL-based regression and uses assigner/sampler from train_cfg.
#     """

#     def __init__(self,
#                  num_classes: int,
#                  in_channels: Sequence[int],
#                  feat_channels: int = 256,
#                  reg_max: int = 16,
#                  stacked_convs: int = 0,
#                  strides: Sequence[int] = (8, 16, 32),
#                  use_depthwise: bool = False,
#                  loss_cls: ConfigDict = dict(
#                      type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#                  loss_bbox: ConfigDict = dict(
#                      type='IoULoss', mode='square', loss_weight=5.0),
#                  loss_obj: ConfigDict = dict(
#                      type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#                  loss_dfl_weight: float = 1.0,
#                  train_cfg: Optional[ConfigDict] = None,
#                  test_cfg: Optional[ConfigDict] = None,
#                  init_cfg: Optional[dict] = None) -> None:
#         super().__init__(init_cfg=init_cfg)

#         self.num_classes = num_classes
#         self.in_channels = in_channels  # list of channels (per level)
#         self.feat_channels = feat_channels
#         self.reg_max = reg_max
#         self.stacked_convs = stacked_convs
#         self.strides = tuple(strides)
#         self.use_depthwise = use_depthwise

#         self.prior_generator = MlvlPointGenerator(self.strides, offset=0.5)

#         # losses
#         self.loss_cls = MODELS.build(loss_cls)
#         self.loss_bbox = MODELS.build(loss_bbox)
#         self.loss_obj = MODELS.build(loss_obj)
#         self.loss_dfl_weight = loss_dfl_weight

#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg

#         if self.train_cfg:
#             # train_cfg should include 'assigner' top-level
#             self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
#             self.sampler = PseudoSampler()

#         # build layers: for each input channel we build cv2 (reg->4*reg_max) and cv3 (cls->nc) and optionally obj conv
#         self.reg_convs = nn.ModuleList()
#         self.cls_convs = nn.ModuleList()
#         self.reg_preds = nn.ModuleList()  # conv to 4*reg_max
#         self.cls_preds = nn.ModuleList()  # conv to num_classes
#         self.obj_preds = nn.ModuleList()  # conv to 1 (objectness scaling)

#         Conv = nn.Conv2d
#         for c in self.in_channels:
#             # optional stacked convs
#             if self.stacked_convs > 0:
#                 reg_layers = []
#                 cls_layers = []
#                 ch = c
#                 for i in range(self.stacked_convs):
#                     if use_depthwise:
#                         reg_layers += [nn.Conv2d(ch, ch, 3, padding=1, groups=ch), nn.ReLU(inplace=True)]
#                         cls_layers += [nn.Conv2d(ch, ch, 3, padding=1, groups=ch), nn.ReLU(inplace=True)]
#                     else:
#                         reg_layers += [nn.Conv2d(ch, self.feat_channels, 3, padding=1), nn.ReLU(inplace=True)]
#                         cls_layers += [nn.Conv2d(ch, self.feat_channels, 3, padding=1), nn.ReLU(inplace=True)]
#                         ch = self.feat_channels
#                 self.reg_convs.append(nn.Sequential(*reg_layers))
#                 self.cls_convs.append(nn.Sequential(*cls_layers))
#             else:
#                 # identity convs (no stacks) - just a small conv to unify channels if needed
#                 self.reg_convs.append(nn.Identity())
#                 self.cls_convs.append(nn.Identity())

#             self.reg_preds.append(nn.Conv2d(c if self.stacked_convs == 0 else self.feat_channels,
#                                             4 * self.reg_max, kernel_size=1))
#             self.cls_preds.append(nn.Conv2d(c if self.stacked_convs == 0 else self.feat_channels,
#                                             self.num_classes, kernel_size=1))
#             self.obj_preds.append(nn.Conv2d(c if self.stacked_convs == 0 else self.feat_channels,
#                                             1, kernel_size=1))

#         # DFL module: produce expected value from distribution logits
#         # Implementation: reshape logits and compute expected value via softmax * arange
#         # We'll implement as functional, not a separate Module.
#         self._init_weights()

#     def _init_weights(self):
#         # init biases for stability
#         bias_init = bias_init_with_prob(0.01)
#         for cls_conv, obj_conv in zip(self.cls_preds, self.obj_preds):
#             cls_conv.bias.data.fill_(bias_init)
#             obj_conv.bias.data.fill_(bias_init)
#         # reg preds default init (no bias change)
#         # other convs: default init

#     def forward_single(self, x: Tensor, reg_conv: nn.Module, cls_conv: nn.Module,
#                        reg_pred: nn.Module, cls_pred: nn.Module, obj_pred: nn.Module):
#         """
#         Single-level forward. Return:
#           - reg_logits: [B, 4*reg_max, H, W]
#           - cls_logits: [B, nc, H, W]
#           - obj_logits: [B, 1, H, W]
#         """
#         if isinstance(reg_conv, nn.Identity):
#             reg_feat = x
#             cls_feat = x
#         else:
#             reg_feat = reg_conv(x)
#             cls_feat = cls_conv(x)
#         reg_logits = reg_pred(reg_feat)
#         cls_logits = cls_pred(cls_feat)
#         obj_logits = obj_pred(reg_feat)
#         return reg_logits, cls_logits, obj_logits

#     def forward(self, feats: Sequence[Tensor]):
#         """Forward on multiple feature levels: returns tuple of lists (reg_logits, cls_logits, obj_logits)"""
#         outs = mmd_multi_apply(self.forward_single, feats, self.reg_convs, self.cls_convs,
#                                self.reg_preds, self.cls_preds, self.obj_preds)
#         # outs is tuple of lists: (list_reg_logits, list_cls_logits, list_obj_logits)
#         return outs

#     def _compute_expected(self, logits: Tensor) -> Tensor:
#         """
#         logits: [N, 4*reg_max, HW] OR [B, 4*reg_max, H, W]
#         return expected distances per side: [N, 4, HW] or [B,4,H,W] depending on input
#         """
#         # If 4*reg_max in channel dim
#         if logits.dim() == 4:
#             b, c, h, w = logits.shape
#             logits = logits.view(b, 4, self.reg_max, h * w).permute(0, 1, 3, 2)  # [B,4,HW,reg_max]
#             probs = F.softmax(logits, dim=-1)
#             weights = torch.arange(self.reg_max, dtype=probs.dtype, device=probs.device).view(1, 1, 1, -1)
#             exp = (probs * weights).sum(-1)  # [B,4,HW]
#             exp = exp.view(b, 4, h, w)
#             return exp
#         else:
#             # logits [N,4,reg_max] -> expected [N,4]
#             probs = F.softmax(logits, dim=-1)
#             weights = torch.arange(self.reg_max, dtype=probs.dtype, device=probs.device).view(1, 1, -1)
#             exp = (probs * weights).sum(-1)
#             return exp

#     def _flatten_outputs(self, reg_logits_list, cls_logits_list, obj_logits_list):
#         """
#         Convert per-level outputs to flattened tensors:
#          - flatten_reg_logits: list of [B, 4*reg_max, H, W] -> per-level permute-> reshape -> [B, N_lvl, 4*reg_max] then concatenated
#          - flatten_cls: [B, N_all, nc]
#          - flatten_obj: [B, N_all]
#         Also return mlvl_priors
#         """
#         device = reg_logits_list[0].device
#         dtype = reg_logits_list[0].dtype
#         num_imgs = reg_logits_list[0].shape[0]
#         featmap_sizes = [r.shape[2:] for r in reg_logits_list]
#         mlvl_priors = self.prior_generator.grid_priors(
#             featmap_sizes,
#             dtype=dtype,
#             device=device,
#             with_stride=True)

#         flatten_reg = []
#         flatten_cls = []
#         flatten_obj = []
#         for reg_logits, cls_logits, obj_logits in zip(reg_logits_list, cls_logits_list, obj_logits_list):
#             b, _, h, w = reg_logits.shape
#             flatten_reg.append(reg_logits.permute(0, 2, 3, 1).reshape(b, -1, 4 * self.reg_max))
#             flatten_cls.append(cls_logits.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes))
#             flatten_obj.append(obj_logits.permute(0, 2, 3, 1).reshape(b, -1))
#         flatten_reg = torch.cat(flatten_reg, dim=1)  # [B, N_all, 4*reg_max]
#         flatten_cls = torch.cat(flatten_cls, dim=1)  # [B, N_all, nc]
#         flatten_obj = torch.cat(flatten_obj, dim=1)  # [B, N_all]
#         flatten_priors = torch.cat(mlvl_priors, dim=0)  # [N_all, 4]
#         return flatten_reg, flatten_cls, flatten_obj, flatten_priors, mlvl_priors

#     def loss_by_feat(self,
#                      reg_logits: Sequence[Tensor],
#                      cls_logits: Sequence[Tensor],
#                      obj_logits: Sequence[Tensor],
#                      batch_gt_instances: Sequence[InstanceData],
#                      batch_img_metas: Sequence[dict],
#                      gt_instances_ignore=None) -> dict:
#         """
#         Compute losses from network outputs.
#         - reg_logits: list of [B, 4*reg_max, H, W]
#         - cls_logits: list of [B, nc, H, W]
#         - obj_logits: list of [B, 1, H, W]
#         """
#         num_imgs = len(batch_gt_instances)
#         if gt_instances_ignore is None:
#             gt_instances_ignore = [None] * num_imgs

#         flatten_reg, flatten_cls, flatten_obj, flatten_priors, mlvl_priors = self._flatten_outputs(
#             reg_logits, cls_logits, obj_logits)

#         # decoded bboxes (from expected values)
#         # compute expected values per side from reg logits
#         # flatten_reg: [B, N_all, 4*reg_max]
#         B, N_all, _ = flatten_reg.shape
#         device = flatten_reg.device
#         # reshape to [B*N_all, 4, reg_max]
#         flat_reg_logits = flatten_reg.view(B * N_all, 4, self.reg_max)
#         exp_values = F.softmax(flat_reg_logits, dim=-1)
#         weights = torch.arange(self.reg_max, dtype=exp_values.dtype, device=device).view(1, 1, -1)
#         expected = (exp_values * weights).sum(-1)  # [B*N_all,4]
#         # expected distances relative to prior scale (units: cells). priors -> [N_all,4]
#         # expand priors to batch dimension
#         priors = flatten_priors.to(device)
#         priors_batch = priors.unsqueeze(0).expand(B, -1, -1).reshape(B * N_all, 4)
#         decoded = dist_to_bbox(expected, priors_batch)  # [B*N_all,4]
#         decoded = decoded.view(B, N_all, 4)

#         # prepare flattened cls/object targets
#         flatten_cls_preds = flatten_cls  # [B, N_all, nc]
#         flatten_obj_preds = flatten_obj  # [B, N_all]

#         # call assigner per image
#         (pos_masks, cls_targets, obj_targets, bbox_targets, dfl_targets,
#          l1_targets, num_pos_imgs) = mmd_multi_apply(
#             self._get_targets_single,
#             priors.unsqueeze(0).expand(B, -1, -1),
#             flatten_cls_preds.detach(),
#             decoded.detach(),
#             flatten_obj_preds.detach(),
#             batch_gt_instances,
#             batch_img_metas,
#             gt_instances_ignore)

#         num_pos = sum(num_pos_imgs)
#         num_total_samples = max(reduce_mean(torch.tensor(float(num_pos)).to(device)), 1.0)

#         pos_masks = torch.cat(pos_masks, dim=0)  # [sum_pos,] boolean over flattened preds
#         cls_targets = torch.cat(cls_targets, dim=0)  # [sum_pos, nc] (one-hot * iou)
#         obj_targets = torch.cat(obj_targets, dim=0)  # [B*N_all, 1]
#         bbox_targets = torch.cat(bbox_targets, dim=0)  # [sum_pos, 4]
#         dfl_targets = torch.cat(dfl_targets, dim=0)  # [sum_pos, 4] float distances (target)
#         if l1_targets is not None:
#             l1_targets = torch.cat(l1_targets, dim=0)

#         # classification loss
#         flatten_cls_preds_all = flatten_cls_preds.view(-1, self.num_classes)
#         if num_pos > 0:
#             loss_cls = self.loss_cls(flatten_cls_preds_all[pos_masks], cls_targets) / num_total_samples
#         else:
#             loss_cls = flatten_cls_preds_all.sum() * 0.0

#         # objectness loss (use flatten_obj_preds)
#         flatten_obj_preds_all = flatten_obj_preds.view(-1, 1)
#         loss_obj = self.loss_obj(flatten_obj_preds_all, obj_targets) / num_total_samples

#         # bbox IoU loss on positive decoded boxes
#         if num_pos > 0:
#             # pick predicted decoded boxes for positives
#             decoded_all = decoded.view(-1, 4)
#             pred_pos_decoded = decoded_all[pos_masks]
#             loss_bbox = self.loss_bbox(pred_pos_decoded, bbox_targets) / num_total_samples
#             # DFL loss for distribution logits (regression distribution)
#             pred_reg_pos_logits = flat_reg_logits[pos_masks]  # [num_pos,4,reg_max]
#             loss_dfl = distribution_focal_loss(pred_reg_pos_logits, dfl_targets, self.reg_max) / num_total_samples
#             loss_dfl = loss_dfl * self.loss_dfl_weight
#         else:
#             loss_bbox = decoded.sum() * 0.0
#             loss_dfl = flat_reg_logits.sum() * 0.0

#         loss_dict = dict(loss_cls=loss_cls, loss_obj=loss_obj, loss_bbox=loss_bbox, loss_dfl=loss_dfl)
#         return loss_dict

#     @torch.no_grad()
#     def _get_targets_single(self,
#                             priors: Tensor,
#                             cls_preds: Tensor,
#                             decoded_bboxes: Tensor,
#                             objectness: Tensor,
#                             gt_instances: InstanceData,
#                             img_meta: dict,
#                             gt_instances_ignore: Optional[InstanceData] = None):
#         """
#         Create targets for one image.
#         priors: [N_all, 4] (cx, cy, w, h)
#         cls_preds: [N_all, nc]
#         decoded_bboxes: [N_all, 4] decoded from expected values
#         objectness: [N_all]
#         gt_instances: InstanceData (bboxes: [M,4], labels: [M])
#         """
#         num_priors = priors.size(0)
#         num_gts = len(gt_instances)
#         if num_gts == 0:
#             cls_target = cls_preds.new_zeros((0, self.num_classes))
#             bbox_target = cls_preds.new_zeros((0, 4))
#             dfl_target = cls_preds.new_zeros((0, 4))
#             l1_target = cls_preds.new_zeros((0, 4))
#             obj_target = cls_preds.new_zeros((num_priors, 1))
#             foreground_mask = cls_preds.new_zeros(num_priors).bool()
#             return foreground_mask, cls_target, obj_target, bbox_target, dfl_target, l1_target, 0

#         # prepare predicted instances structure for assigner
#         # use decoded_bboxes (tl,tl,br,br) and cls scores * obj sqrt as score (mimic YOLOX)
#         scores = cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid()
#         pred_bboxes = decoded_bboxes
#         pred_scores = scores
#         gt_bboxes = gt_instances.bboxes
#         gt_labels = gt_instances.labels
#         gt_bboxes_ignore = None if gt_instances_ignore is None else gt_instances_ignore.bboxes

#         assign_result = self.assigner(
#             pred_bboxes=pred_bboxes,
#             pred_scores=pred_scores,
#             gt_bboxes=gt_bboxes,
#             gt_labels=gt_labels,
#             #gt_bboxes_ignore=gt_bboxes_ignore
#         )

#         # pred_instances = InstanceData(bboxes=decoded_bboxes, scores=scores)

#         # assign_result = self.assigner(pred_instances=pred_instances,
#         #                                      gt_instances=gt_instances,
#         #                                      gt_instances_ignore=gt_instances_ignore)
#         sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)
#         pos_inds = sampling_result.pos_inds
#         num_pos = pos_inds.size(0)

#         pos_ious = assign_result.max_overlaps[pos_inds] if num_pos > 0 else cls_preds.new_tensor(0.0)
#         cls_target = F.one_hot(sampling_result.pos_gt_labels, self.num_classes).float() * pos_ious.unsqueeze(-1) \
#             if num_pos > 0 else cls_preds.new_zeros((0, self.num_classes))
#         obj_target = cls_preds.new_zeros((num_priors, 1))
#         if num_pos > 0:
#             obj_target[pos_inds] = 1.0
#             bbox_target = sampling_result.pos_gt_bboxes  # [num_pos,4] xyxy
#             # compute dfl targets: distances from prior center to gt (l,t,r,b) in same scale as priors (priors[:,2:4] are stride widths)
#             priors_pos = priors[pos_inds]  # [num_pos,4]
#             gt_cxcywh = bbox_xyxy_to_cxcywh(bbox_target)
#             # compute distances in absolute coordinate, but priors' 3:4 are the cell stride? here priors are (cx,cy,stride_w,stride_h)
#             # so we compute distances in the same absolute coordinate: left = cx - x1, etc.
#             l = gt_cxcywh[:, 0] - (priors_pos[:, 0] - 0.5 * priors_pos[:, 2])
#             t = gt_cxcywh[:, 1] - (priors_pos[:, 1] - 0.5 * priors_pos[:, 3])
#             r = (priors_pos[:, 0] + 0.5 * priors_pos[:, 2]) - gt_cxcywh[:, 0]
#             b = (priors_pos[:, 1] + 0.5 * priors_pos[:, 3]) - gt_cxcywh[:, 1]
#             # ensure >=0
#             dfl_target = torch.stack([l.clamp(min=0.0), t.clamp(min=0.0), r.clamp(min=0.0), b.clamp(min=0.0)], dim=-1)
#             # Optionally scale targets relative to stride (if you want expected to be in cell units)
#             # Here we leave absolute units consistent with priors.
#             l1_target = cls_preds.new_zeros((num_pos, 4))
#             # if you want an l1 target in normalized form:
#             # convert gt to cxcywh and compute offsets relative to priors as in YOLOX
#             gt_cxcywh = bbox_xyxy_to_cxcywh(bbox_target)
#             l1_target[:, :2] = (gt_cxcywh[:, :2] - priors_pos[:, :2]) / priors_pos[:, 2:]
#             l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors_pos[:, 2:] + 1e-8)
#             foreground_mask = cls_preds.new_zeros(num_priors).to(torch.bool)
#             foreground_mask[pos_inds] = 1
#             return foreground_mask, cls_target, obj_target, bbox_target, dfl_target, l1_target, num_pos
#         else:
#             bbox_target = cls_preds.new_zeros((0, 4))
#             dfl_target = cls_preds.new_zeros((0, 4))
#             l1_target = cls_preds.new_zeros((0, 4))
#             foreground_mask = cls_preds.new_zeros(num_priors).bool()
#             return foreground_mask, cls_target, obj_target, bbox_target, dfl_target, l1_target, 0

#     def predict_by_feat(self,
#                         reg_logits: List[Tensor],
#                         cls_logits: List[Tensor],
#                         obj_logits: List[Tensor],
#                         batch_img_metas: Optional[List[dict]] = None,
#                         cfg: Optional[ConfigDict] = None,
#                         rescale: bool = False,
#                         with_nms: bool = True) -> List[InstanceData]:
#         """
#         Decode network outputs into detection results (with NMS).
#         reg_logits: list per level [B, 4*reg_max, H, W]
#         cls_logits: list per level [B, nc, H, W]
#         obj_logits: list per level [B, 1, H, W]
#         """
#         cfg = self.test_cfg if cfg is None else cfg
#         B = cls_logits[0].shape[0]
#         device = cls_logits[0].device
#         featmap_sizes = [c.shape[2:] for c in cls_logits]
#         mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=cls_logits[0].dtype, device=device, with_stride=True)

#         # for each image decode and NMS
#         results = []
#         for img_id in range(B):
#             per_img_bboxes = []
#             per_img_scores = []
#             per_img_labels = []
#             for reg_l, cls_l, obj_l, priors in zip(reg_logits, cls_logits, obj_logits, mlvl_priors):
#                 # reg_l: [B, 4*reg_max, H, W] -> take img slice
#                 reg_img = reg_l[img_id].unsqueeze(0)  # [1, 4*reg_max, H, W]
#                 cls_img = cls_l[img_id].permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()  # [N_lvl, nc]
#                 obj_img = obj_l[img_id].permute(1, 2, 0).reshape(-1).sigmoid()  # [N_lvl]
#                 # expected values
#                 reg_logits_flat = reg_img.view(1, 4, self.reg_max, -1).permute(0, 1, 3, 2)  # [1,4,N,reg_max]
#                 probs = F.softmax(reg_logits_flat, dim=-1)
#                 weights = torch.arange(self.reg_max, dtype=probs.dtype, device=probs.device).view(1, 1, 1, -1)
#                 expected = (probs * weights).sum(-1).squeeze(0)  # [4, N]
#                 expected = expected.permute(1, 0)  # [N,4]
#                 # decode
#                 pri = priors.view(-1, 4).to(device)
#                 bboxes = dist_to_bbox(expected, pri)  # [N,4]
#                 scores_all = cls_img * obj_img.unsqueeze(-1)  # [N, nc]
#                 max_scores, labels = scores_all.max(dim=1)
#                 keep_mask = max_scores > cfg.score_thr
#                 if keep_mask.sum() == 0:
#                     continue
#                 bboxes_keep = bboxes[keep_mask]
#                 scores_keep = max_scores[keep_mask]
#                 labels_keep = labels[keep_mask]
#                 per_img_bboxes.append(bboxes_keep)
#                 per_img_scores.append(scores_keep)
#                 per_img_labels.append(labels_keep)
#             if len(per_img_bboxes) == 0:
#                 results.append(InstanceData(bboxes=cls_logits[0].new_zeros((0, 4)),
#                                             scores=cls_logits[0].new_zeros((0,)),
#                                             labels=cls_logits[0].new_zeros((0,), dtype=torch.long)))
#                 continue
#             bboxes_all = torch.cat(per_img_bboxes, dim=0)
#             scores_all = torch.cat(per_img_scores, dim=0)
#             labels_all = torch.cat(per_img_labels, dim=0)
#             # batched_nms expects boxes [x1,y1,x2,y2], scores, labels, and cfg.nms
#             det_bboxes, keep_idxs = torch.ops.mmcv.batched_nms(bboxes_all, scores_all, labels_all, cfg.nms)
#             # wrap into InstanceData
#             det_scores = det_bboxes[:, -1] if det_bboxes.numel() > 0 and det_bboxes.shape[1] > 4 else scores_all[keep_idxs]
#             # here det_bboxes may contain scores in last column in some ops; handle generically
#             if det_bboxes.shape[1] > 4:
#                 det_bboxes_xyxy = det_bboxes[:, :4]
#                 det_scores = det_bboxes[:, -1]
#             else:
#                 det_bboxes_xyxy = det_bboxes
#             results.append(InstanceData(bboxes=det_bboxes_xyxy, scores=det_scores, labels=labels_all[keep_idxs]))
#         return results
