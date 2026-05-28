"""Residual Loss (L_R) for Phase 2 SSL training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_z_A(z_patch: torch.Tensor, prototypes: torch.Tensor, tau_s: float) -> torch.Tensor:
    """Prototype soft-assignment weighted reconstruction.

    Args:
        z_patch: (..., d) — L2-normalized patch embeddings
        prototypes: (K, d) — L2-normalized prototype matrix
        tau_s: student temperature

    Returns:
        z_A: (..., d) — prototype-reconstructed vector (NOT normalized)
    """
    scores = torch.matmul(z_patch, prototypes.T) / tau_s  # (..., K)
    p = F.softmax(scores, dim=-1)                          # (..., K)
    return torch.matmul(p, prototypes)                     # (..., d)


def compute_z_R(z_patch: torch.Tensor, z_A: torch.Tensor) -> torch.Tensor:
    """Compute residual component orthogonal to z_A.

    Subtracts the projection of z_patch onto z_A (with stop-gradient on z_A).

    Args:
        z_patch: (..., d)
        z_A: (..., d) — prototype reconstruction, used with stop-gradient

    Returns:
        z_R: (..., d)
    """
    z_A_sg = z_A.detach()
    norm_sq = (z_A_sg * z_A_sg).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    proj = (z_patch * z_A_sg).sum(dim=-1, keepdim=True) / norm_sq * z_A_sg
    return z_patch - proj


@torch.no_grad()
def compute_dynamic_weights(per_patch_ce: torch.Tensor, tau_w: float = 0.5) -> torch.Tensor:
    """Dynamic per-patch weights based on iBOT reconstruction error.

    Patches with higher reconstruction error (potential lesions) get larger weights.

    Args:
        per_patch_ce: (M,) — positive CE values (higher = harder patch)
        tau_w: temperature controlling weight sharpness

    Returns:
        w: (M,) — normalized weights, sum ≈ M (each weight is relative to mean)
    """
    scaled = per_patch_ce / tau_w
    exp_vals = torch.exp(scaled - scaled.max())  # numerically stable
    mean_exp = exp_vals.mean().clamp(min=1e-8)
    return exp_vals / mean_exp


class ResidualLoss(nn.Module):
    """Compute L_R — cosine similarity between student and teacher z_R on masked patches.

    Uses dynamic weighting (per_patch_ibot_loss) after a warmup period.

    Args:
        warmup_epochs: number of epochs before dynamic weighting is activated
        tau_w: temperature for dynamic weighting
        tau_s: student temperature for z_A computation
    """

    def __init__(self, warmup_epochs: int = 20, tau_w: float = 0.5, tau_s: float = 0.1):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.tau_w = tau_w
        self.tau_s = tau_s

    def forward(
        self,
        student_z_patch_masked: torch.Tensor,   # (M, d) — student raw backbone embeddings at masked positions
        teacher_z_patch_masked: torch.Tensor,   # (M, d) — teacher raw backbone embeddings at masked positions
        prototypes: torch.Tensor,               # (K, d) — L2-normalized prototype matrix
        per_patch_ibot_ce: torch.Tensor,        # (M,) — per-patch iBOT CE (detached)
        current_epoch: int,
    ) -> torch.Tensor:
        """
        Args:
            student_z_patch_masked: (M, d) — patch embeddings at masked positions (student)
            teacher_z_patch_masked: (M, d) — patch embeddings at masked positions (teacher)
            prototypes: (K, d)
            per_patch_ibot_ce: (M,) — from iBOTPatchLoss.forward_masked(return_per_patch=True)
            current_epoch: for warmup gate

        Returns:
            L_R scalar
        """
        if student_z_patch_masked.shape[0] == 0:
            return torch.tensor(0.0, device=student_z_patch_masked.device)

        # L2 normalize patches (cast to float32 — backbone may output bf16)
        s_norm = F.normalize(student_z_patch_masked.float(), dim=-1)  # (M, d)
        t_norm = F.normalize(teacher_z_patch_masked.float(), dim=-1)  # (M, d)

        # Compute z_A and z_R for both student and teacher
        z_A_s = compute_z_A(s_norm, prototypes.detach().float(), self.tau_s)   # (M, d)
        z_A_t = compute_z_A(t_norm, prototypes.detach().float(), self.tau_s)   # (M, d)

        z_R_s = compute_z_R(s_norm, z_A_s)   # (M, d)
        z_R_t = compute_z_R(t_norm, z_A_t)   # (M, d)

        # Guard against z_R collapse: skip patches where either z_R is near-zero
        z_R_s_norm_val = z_R_s.norm(dim=-1)   # (M,)
        z_R_t_norm_val = z_R_t.norm(dim=-1)   # (M,)
        valid = (z_R_s_norm_val > 1e-4) & (z_R_t_norm_val > 1e-4)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=student_z_patch_masked.device)

        z_R_s = z_R_s[valid]
        z_R_t = z_R_t[valid]
        if per_patch_ibot_ce is not None:
            per_patch_ibot_ce = per_patch_ibot_ce[valid]

        # Cosine similarity loss: 1 - cos_sim (in [0, 2])
        z_R_s_norm = F.normalize(z_R_s, dim=-1)
        z_R_t_norm = F.normalize(z_R_t.detach(), dim=-1)  # stop-grad on teacher

        # per-patch cosine loss: 1 - cos_sim (in [0, 2])
        per_patch_lr = 1.0 - (z_R_s_norm * z_R_t_norm).sum(dim=-1)  # (M,)

        # Dynamic weighting (activated after warmup)
        if current_epoch >= self.warmup_epochs and per_patch_ibot_ce is not None:
            weights = compute_dynamic_weights(per_patch_ibot_ce, self.tau_w)  # (M,)
            loss = (per_patch_lr * weights).mean()
        else:
            loss = per_patch_lr.mean()

        return loss
