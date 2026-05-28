"""Phase 1 SSL Architecture: iBOT + Prototype Assignment Loss (L_A)."""

from functools import partial
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from chexfound.loss import iBOTPatchLoss, PrototypeDictionary, PrototypeQueue, PrototypeAssignmentLoss
from chexfound.models import build_model_from_cfg
from chexfound.layers import DINOHead
from chexfound.utils.utils import has_batchnorms
from chexfound.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from chexfound.fsdp import get_fsdp_wrapper, ShardedGradScaler, reshard_fsdp_model
from chexfound.models.vision_transformer import BlockChunk
import chexfound.distributed as distributed


try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("chexfound")


class SSLPhase1Arch(nn.Module):
    """Phase 1: iBOT patch reconstruction + Prototype Assignment (L_A).

    No DINO CLS loss, no KoLeo. Only backbone + ibot_head.
    PrototypeDictionary is NOT FSDP-wrapped; gradients are manually all-reduced.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        self.embed_dim = embed_dim

        ibot_head_factory = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.ibot.head_n_prototypes,
            hidden_dim=cfg.ibot.head_hidden_dim,
            bottleneck_dim=cfg.ibot.head_bottleneck_dim,
            nlayers=cfg.ibot.head_nlayers,
        )

        self.student = nn.ModuleDict({
            "backbone": student_backbone,
            "ibot_head": ibot_head_factory(),
        })
        self.teacher = nn.ModuleDict({
            "backbone": teacher_backbone,
            "ibot_head": ibot_head_factory(),
        })
        for p in self.teacher.parameters():
            p.requires_grad = False

        # iBOT patch loss
        self.ibot_loss_weight = cfg.ibot.loss_weight
        self.ibot_patch_loss = iBOTPatchLoss(cfg.ibot.head_n_prototypes)

        # Prototype components
        proto_cfg = cfg.prototype
        self.prototype_dict = PrototypeDictionary(K=proto_cfg.K, d=proto_cfg.d)
        self.queue = PrototypeQueue(
            queue_size=proto_cfg.queue_size,
            d=proto_cfg.d,
            device=torch.device("cpu"),  # moved to cuda in forward
        )
        self.prototype_loss_fn = PrototypeAssignmentLoss(
            tau_t=proto_cfg.tau_t,
            tau_s=proto_cfg.tau_s,
            sinkhorn_iters=proto_cfg.sinkhorn_iters,
            patch_size=cfg.student.patch_size,
        )
        self.lambda_A = proto_cfg.lambda_A

        self.need_to_synchronize_fsdp_streams = True
        logger.info(f"SSLPhase1Arch built: embed_dim={embed_dim}, K={proto_cfg.K}, queue={proto_cfg.queue_size}")

    @torch.no_grad()
    def init_prototypes_from_pca(self, data):
        """Teacher 첫 배치 z_patch로 PCA prototype 초기화. rank 0에서 계산 후 broadcast.

        reshard_fsdp_model을 호출하지 않는다 — 첫 forward 이전에는 FSDP _local_shard가
        아직 초기화되지 않아 AttributeError가 발생한다. 메모리는 이후 첫 학습 step에서
        자동으로 정리된다.
        """
        global_crops = data["collated_global_crops"].cuda(non_blocking=True)
        teacher_out = self.teacher.backbone(global_crops, is_training=True)
        z_patch = teacher_out["x_norm_patchtokens"]  # (2B, N_g, d)
        t_norm = F.normalize(z_patch.flatten(0, 1).float(), dim=-1)
        if distributed.is_main_process():
            self.prototype_dict.init_from_pca(t_norm)
        if distributed.is_enabled():
            for p in self.prototype_dict.parameters():
                dist.broadcast(p.data, src=0)

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()
        return True

    def forward_backward(self, images, teacher_temp, lambda_A=None):
        n_global_crops = 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)
        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)
        crop_params_batch = images["crop_params"]  # CPU list, not moved to CUDA

        B = global_crops.shape[0] // n_global_crops

        # --- Teacher forward (no grad) ---
        @torch.no_grad()
        def get_teacher_output():
            teacher_out = self.teacher.backbone(global_crops, is_training=True)
            teacher_z_patch_all = teacher_out["x_norm_patchtokens"]  # (2B, N_g, d)
            _dim = teacher_z_patch_all.shape[-1]

            buffer_teacher = teacher_z_patch_all.new_zeros(upperbound, _dim)
            torch.index_select(
                teacher_z_patch_all.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_teacher[:n_masked_patches],
            )
            masked_teacher_after_head = self.teacher.ibot_head(buffer_teacher)[:n_masked_patches]
            masked_teacher_softmaxed = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                masked_teacher_after_head,
                teacher_temp=teacher_temp,
                n_masked_patches_tensor=n_masked_patches_tensor,
            )
            return teacher_z_patch_all, masked_teacher_softmaxed

        teacher_z_patch, masked_teacher_ibot_softmaxed = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        # --- Student forward ---
        student_global_out, student_local_out = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )
        student_z_patch_global = student_global_out["x_norm_patchtokens"]  # (2B, N_g, d)
        student_z_patch_local = student_local_out["x_norm_patchtokens"]    # (n_local*B, N_l, d)

        _dim = student_z_patch_global.shape[-1]
        buffer_student = student_z_patch_global.new_zeros(upperbound, _dim)
        buffer_student[:n_masked_patches].copy_(
            torch.index_select(student_z_patch_global.flatten(0, 1), dim=0, index=mask_indices_list)
        )
        student_masked_after_head = self.student.ibot_head(buffer_student)[:n_masked_patches]

        loss_dict = {}
        loss_accumulator = torch.tensor(0.0, device=global_crops.device)

        # --- L_iBOT ---
        ibot_loss = (
            self.ibot_patch_loss.forward_masked(
                student_masked_after_head,
                masked_teacher_ibot_softmaxed,
                student_masks_flat=masks,
                n_masked_patches=n_masked_patches,
                masks_weight=masks_weight,
            )
            * 2  # loss_scales = n_global_crops
        )
        loss_dict["ibot_loss"] = ibot_loss / 2
        loss_accumulator = loss_accumulator + self.ibot_loss_weight * ibot_loss

        # --- L_A (Prototype Assignment) ---
        # Move queue buffer to GPU on first use
        if self.queue.buffer.device.type == "cpu":
            self.queue.buffer = self.queue.buffer.to(global_crops.device)

        la_loss = self.prototype_loss_fn(
            teacher_z_patch=teacher_z_patch,
            student_z_patch_global=student_z_patch_global,
            student_z_patch_local=student_z_patch_local,
            crop_params_batch=crop_params_batch,
            prototype_dict=self.prototype_dict,
            queue=self.queue,
        )
        loss_dict["la_loss"] = la_loss
        _lambda_A = lambda_A if lambda_A is not None else self.lambda_A
        loss_accumulator = loss_accumulator + _lambda_A * la_loss

        did_backward = self.backprop_loss(loss_accumulator)
        if not did_backward:
            # Poison check: return NaN sentinel so train loop can skip optimizer.step()
            return {k: torch.tensor(float('nan'), device=global_crops.device) for k in loss_dict}
        self.fsdp_synchronize_streams()

        return loss_dict

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.ibot_head._streams = (
                self.teacher.ibot_head._streams
            ) = self.student.backbone._streams = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        with torch.no_grad():
            for k in self.student.keys():
                student_params = list(self.student[k].parameters())
                teacher_params = list(self.teacher[k].parameters())
                torch._foreach_mul_(teacher_params, m)
                torch._foreach_add_(teacher_params, student_params, alpha=1 - m)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused = fuse_params_groups(params_groups)
        for g in fused:
            g["foreach"] = True
        return fused

    def get_params_groups(self):
        # Student backbone + ibot_head with layerwise decay
        all_groups = []
        for m in self.student.values():
            all_groups += self.get_maybe_fused_params_for_submodel(m)
        # PrototypeDictionary: single param group, no layerwise decay
        all_groups.append({
            "params": list(self.prototype_dict.parameters()),
            "lr_multiplier": 1.0,
            "wd_multiplier": 1.0,
            "is_last_layer": False,
            "foreach": True,
        })
        return all_groups

    def prepare_for_distributed_training(self):
        logger.info("FSDP -- preparing Phase 1 model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        for k in self.student.keys():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
        # Broadcast prototype_dict from rank 0 to all ranks (not FSDP-wrapped)
        if distributed.is_enabled():
            for p in self.prototype_dict.parameters():
                dist.broadcast(p.data, src=0)
        logger.info("Phase 1 distributed training preparation complete.")
