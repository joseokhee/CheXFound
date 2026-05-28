"""Phase 2 SSL Architecture: iBOT + L_A (0.1) + Residual Loss (L_R) with dynamic weighting."""

import logging

import torch
import torch.nn.functional as F

from chexfound.loss import ResidualLoss
from chexfound.train.cxr_ssl_phase1_arch import SSLPhase1Arch
from chexfound.fsdp import reshard_fsdp_model
import chexfound.distributed as distributed


logger = logging.getLogger("chexfound")


class SSLPhase2Arch(SSLPhase1Arch):
    """Phase 2 extends Phase 1 with a frozen prototype dict and residual loss L_R.

    PrototypeDictionary is loaded from Phase 1 checkpoint and frozen.
    ResidualLoss uses dynamic weighting based on per-patch iBOT reconstruction error.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        res_cfg = cfg.residual
        self.residual_loss_fn = ResidualLoss(
            warmup_epochs=res_cfg.warmup_epochs,
            tau_w=res_cfg.tau_w,
            tau_s=cfg.prototype.tau_s,
        )
        self.lambda_R = res_cfg.lambda_R
        self.lambda_A_p2 = res_cfg.lambda_A_P2
        self.lambda_A = self.lambda_A_p2  # override Phase 1 lambda_A

        logger.info(
            f"SSLPhase2Arch: lambda_R={self.lambda_R}, lambda_A_P2={self.lambda_A_p2}, "
            f"residual warmup={res_cfg.warmup_epochs}"
        )

    def freeze_prototype_dict(self):
        """Freeze prototype dictionary (called after loading Phase 1 checkpoint)."""
        for p in self.prototype_dict.parameters():
            p.requires_grad = False
        logger.info("PrototypeDictionary frozen for Phase 2.")

    def forward_backward(self, images, teacher_temp, current_epoch: int = 0):
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
        crop_params_batch = images["crop_params"]

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
            # Raw masked patch embeddings for L_R (before ibot_head)
            teacher_z_patch_masked = torch.index_select(
                teacher_z_patch_all.flatten(0, 1), dim=0, index=mask_indices_list
            )  # (n_masked_patches, d)
            return teacher_z_patch_all, masked_teacher_softmaxed, teacher_z_patch_masked

        teacher_z_patch, masked_teacher_ibot_softmaxed, teacher_z_masked = get_teacher_output()
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

        # Raw masked student embeddings for L_R
        student_z_masked = torch.index_select(
            student_z_patch_global.flatten(0, 1), dim=0, index=mask_indices_list
        )  # (n_masked_patches, d)

        loss_dict = {}
        loss_accumulator = torch.tensor(0.0, device=global_crops.device)

        # --- L_iBOT (with per-patch loss for dynamic weighting) ---
        ibot_loss, per_patch_ce = self.ibot_patch_loss.forward_masked(
            student_masked_after_head,
            masked_teacher_ibot_softmaxed,
            student_masks_flat=masks,
            n_masked_patches=n_masked_patches,
            masks_weight=masks_weight,
            return_per_patch=True,
        )
        ibot_loss = ibot_loss * 2  # loss_scales
        loss_dict["ibot_loss"] = ibot_loss / 2
        loss_accumulator = loss_accumulator + self.ibot_loss_weight * ibot_loss

        # --- L_A (Prototype Assignment, λ_A = 0.1) ---
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
        loss_accumulator = loss_accumulator + self.lambda_A_p2 * la_loss

        # --- L_R (Residual Loss) ---
        if n_masked_patches > 0:
            lr_loss = self.residual_loss_fn(
                student_z_patch_masked=student_z_masked[:n_masked_patches],
                teacher_z_patch_masked=teacher_z_masked[:n_masked_patches],
                prototypes=self.prototype_dict.prototypes,
                per_patch_ibot_ce=per_patch_ce[:n_masked_patches],
                current_epoch=current_epoch,
            )
        else:
            lr_loss = torch.tensor(0.0, device=global_crops.device)

        loss_dict["lr_loss"] = lr_loss
        loss_accumulator = loss_accumulator + self.lambda_R * lr_loss

        self.backprop_loss(loss_accumulator)
        self.fsdp_synchronize_streams()

        return loss_dict
