"""Prototype Dictionary, Sinkhorn-Knopp Queue, and Prototype Assignment Loss (L_A)."""

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from chexfound.data.augmentations_params import compute_overlap_patch_indices


logger = logging.getLogger("chexfound")


# ---------------------------------------------------------------------------
# Sinkhorn-Knopp (log-domain, numerically stable)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sinkhorn_knopp_log(scores, num_iters=3):
    """Log-domain Sinkhorn-Knopp for uniform prototype assignment.

    Args:
        scores: (N, K) — raw log-scores (dot-products / tau), NOT yet softmaxed
    Returns:
        Q: (N, K) — soft assignment probabilities, rows sum to 1/N * N = 1 per row
                    after the final *N scaling rows sum to 1 (each row is a prob dist).
    """
    Q = scores.T.float()  # (K, N)
    K_dim, N = Q.shape

    for _ in range(num_iters):
        # Row normalization: each prototype used equally
        Q -= torch.logsumexp(Q, dim=1, keepdim=True)
        Q -= torch.log(torch.tensor(K_dim, dtype=Q.dtype, device=Q.device))
        # Column normalization: each sample assigns total prob 1
        Q -= torch.logsumexp(Q, dim=0, keepdim=True)
        Q -= torch.log(torch.tensor(N, dtype=Q.dtype, device=Q.device))

    Q = Q.T  # (N, K)
    # Scale by N so each row (patch) sums to ~1 — proper probability distribution
    # Without this, rows sum to 1/N, making la_loss N× too small (effectively kills L_A)
    return torch.exp(Q) * N


# ---------------------------------------------------------------------------
# PrototypeDictionary
# ---------------------------------------------------------------------------

class PrototypeDictionary(nn.Module):
    """Learnable prototype matrix C ∈ R^{K×d}, constrained to the unit sphere.

    Prototypes are L2-normalized after each optimizer step.
    """

    def __init__(self, K: int, d: int):
        super().__init__()
        self.K = K
        self.d = d
        # Initialize on unit sphere
        prototypes = torch.randn(K, d)
        prototypes = F.normalize(prototypes, dim=1)
        self.prototypes = nn.Parameter(prototypes)

    @torch.no_grad()
    def normalize(self):
        """Call after every optimizer step to project back onto the unit sphere."""
        self.prototypes.data = F.normalize(self.prototypes.data, dim=1)

    def forward(self, z_patch):
        """Compute cosine dot-product scores.

        Args:
            z_patch: (..., d) — should be L2-normalized before calling

        Returns:
            scores: (..., K)
        """
        return torch.matmul(z_patch, self.prototypes.T)

    def load_kmeans_init(self, path: str):
        """Load k-means cluster centres from a .npy file."""
        import numpy as np
        centres = torch.from_numpy(np.load(path)).float()  # (K, d)
        centres = F.normalize(centres, dim=1)
        with torch.no_grad():
            self.prototypes.copy_(centres)
        logger.info(f"Loaded k-means prototype init from {path}")

    @torch.no_grad()
    def init_from_pca(self, z_patch: torch.Tensor):
        """Initialize prototypes from top-K PCA directions of z_patch.

        Much better than k-means init because:
        - PCA components are orthogonal by construction (pairwise sim = 0)
        - They capture the axes of maximum variance in actual CXR patch space
        - Requires only one batch of patches (no separate preprocessing step)
        - Stable with as few as ~1000 patches in 1024-dim

        Args:
            z_patch: (N, d) — L2-normalized teacher patch tokens from the first batch.
                     Should have N >> K (e.g. 73K patches from batch=54 is plenty).
        """
        z = z_patch.float()  # (N, d)
        # Subsample to K*10 patches max — PCA directions are stable with ~5000 vectors
        max_pts = self.K * 10
        if z.shape[0] > max_pts:
            idx = torch.randperm(z.shape[0], device=z.device)[:max_pts]
            z = z[idx]
        # Move to GPU for fast SVD (stays on whatever device z_patch was on)
        # Center (mean subtraction on unit sphere approximation)
        z = z - z.mean(dim=0, keepdim=True)
        # SVD: top-K right singular vectors = principal directions
        _, _, Vh = torch.linalg.svd(z, full_matrices=False)  # Vh: (min(N,d), d)
        pca_dirs = Vh[:self.K]  # (K, d)
        pca_dirs = F.normalize(pca_dirs, dim=1)
        self.prototypes.data.copy_(pca_dirs)
        # Verify diversity
        sub = pca_dirs[:min(64, self.K)]
        sim = (sub @ sub.T)
        mask = ~torch.eye(sub.shape[0], dtype=torch.bool, device=sub.device)
        mean_sim = sim[mask].mean().item()
        logger.info(
            f"PCA prototype init: top-{self.K} directions from {z_patch.shape[0]} patches, "
            f"pairwise sim={mean_sim:.4f} (ideal≈0)"
        )


# ---------------------------------------------------------------------------
# PrototypeQueue  (per-GPU local FIFO queue, fp16 storage)
# ---------------------------------------------------------------------------

class PrototypeQueue:
    """FIFO queue of Teacher z_patch embeddings (L2-normalized, fp16 storage).

    Each GPU maintains its own local queue.  The Sinkhorn is computed over
    the local batch + local queue, providing stable assignment for the local
    distribution.  No cross-GPU communication is required for the queue.

    Args:
        queue_size: total number of patch vectors to store
        d: embedding dimension (1024 for ViT-L)
        device: torch device
    """

    def __init__(self, queue_size: int, d: int, device: torch.device):
        self.queue_size = queue_size
        self.d = d
        # fp16 for memory efficiency
        self.buffer = torch.zeros(queue_size, d, dtype=torch.float16, device=device)
        self.ptr = 0          # next write position
        self.fill_count = 0   # how many slots have been written at least once

    def is_ready(self, min_fill: int = 1000) -> bool:
        return self.fill_count >= min_fill

    @torch.no_grad()
    def enqueue(self, z: torch.Tensor):
        """Add L2-normalized teacher patch tokens to the FIFO queue.

        Args:
            z: (N, d) float32 tensor — L2-normalized Teacher z_patch tokens.
               Caller should pass t_norm (all patch tokens, shape 2B*N_g, d).
               Queue covers ~queue_size / (2*B*N_g) steps of patch history.
               Same distribution as current batch → stable Sinkhorn context.
        """
        z_fp16 = z.detach().half()
        n = z_fp16.shape[0]

        if n >= self.queue_size:
            # Entire queue replaced; just keep the last queue_size entries
            self.buffer.copy_(z_fp16[-self.queue_size:])
            self.ptr = 0
            self.fill_count = self.queue_size
            return

        space = self.queue_size - self.ptr
        if n <= space:
            self.buffer[self.ptr: self.ptr + n] = z_fp16
            self.ptr = (self.ptr + n) % self.queue_size
        else:
            # Wrap around
            self.buffer[self.ptr:] = z_fp16[:space]
            self.buffer[:n - space] = z_fp16[space:]
            self.ptr = n - space

        self.fill_count = min(self.fill_count + n, self.queue_size)

    @torch.no_grad()
    def get(self) -> torch.Tensor:
        """Return valid queue content as float32."""
        valid = min(self.fill_count, self.queue_size)
        return self.buffer[:valid].float()


# ---------------------------------------------------------------------------
# Prototype Assignment Loss  (L_A)
# ---------------------------------------------------------------------------

class PrototypeAssignmentLoss(nn.Module):
    """Compute L_A — Sinkhorn prototype assignment cross-entropy.

    For each (teacher_global_view, student_view) pair, overlapping patches are
    found and cross-entropy(Sinkhorn_teacher_target, softmax_student_prediction)
    is computed.

    Args:
        tau_t: teacher temperature for Sinkhorn (sharp targets, default 0.04)
        tau_s: student temperature for softmax predictions (default 0.1)
        sinkhorn_iters: number of Sinkhorn iterations (default 3)
        patch_size: ViT patch size in pixels (default 14)
    """

    def __init__(self, tau_t: float = 0.04, tau_s: float = 0.1,
                 sinkhorn_iters: int = 3, patch_size: int = 14):
        super().__init__()
        self.tau_t = tau_t
        self.tau_s = tau_s
        self.sinkhorn_iters = sinkhorn_iters
        self.patch_size = patch_size

    def forward(
        self,
        teacher_z_patch,        # (2B, N_g, d)
        student_z_patch_global,  # (2B, N_g, d)
        student_z_patch_local,   # (n_local*B, N_l, d)
        crop_params_batch,       # list of B items; each = list of (2+n_local) dicts
        prototype_dict: PrototypeDictionary,
        queue: PrototypeQueue,
    ) -> torch.Tensor:
        B = teacher_z_patch.shape[0] // 2
        N_g = teacher_z_patch.shape[1]
        n_local = student_z_patch_local.shape[0] // B
        N_l = student_z_patch_local.shape[1]
        device = teacher_z_patch.device

        # --- L2 normalize (cast to float32 — prototype dict is fp32, backbone output may be bf16) ---
        # Teacher: stop-gradient already (computed under torch.no_grad())
        t_norm = F.normalize(teacher_z_patch.flatten(0, 1).float(), dim=-1)          # (2B*N_g, d)
        # Student: detach so L_A gradient flows only into prototype_dict, NOT into backbone.
        # Without detach, L_A competes with iBOT for backbone updates and la_loss is stuck at log(K).
        sg_norm = F.normalize(student_z_patch_global.detach().flatten(0, 1).float(), dim=-1)  # (2B*N_g, d)
        sl_norm = F.normalize(student_z_patch_local.detach().flatten(0, 1).float(), dim=-1)   # (n_local*B*N_l, d)

        proto = prototype_dict.prototypes.float()  # (K, d)

        # --- Teacher scores + Sinkhorn (with queue) ---
        # Queue stores per-image mean vectors (2B/step) so it actually covers history.
        # Sinkhorn context = current patches + queue mean vectors (as soft anchors).
        if queue.is_ready():
            q_z = queue.get().to(device)           # (Q, d) fp32 — per-image means
            all_z = torch.cat([t_norm, q_z], dim=0)
        else:
            all_z = t_norm

        # float32 필수 — tau_t=0.04로 나누면 25배 증폭, bf16(max≈65504)에서 overflow 위험
        all_scores = torch.mm(all_z.float(), proto.float().T) / self.tau_t  # (N_total, K)
        q_all = sinkhorn_knopp_log(all_scores, num_iters=self.sinkhorn_iters)  # (N_total, K)
        q_teacher_flat = q_all[:t_norm.shape[0]]             # (2B*N_g, K)

        # Enqueue all Teacher patch tokens (2B*N_g vectors) — same distribution as current batch.
        # Design doc §3.4: queue stores Teacher z_patch (stop-gradient), not image means.
        # Image means have different statistics from patch tokens and bias Sinkhorn assignment.
        queue.enqueue(t_norm)

        # --- Student scores ---
        sg_scores = torch.mm(sg_norm, proto.T)  # (2B*N_g, K)
        sl_scores = torch.mm(sl_norm, proto.T)  # (n_local*B*N_l, K)

        # Reshape for easy indexing: [view, B, N, K]
        q_t = q_teacher_flat.view(2, B, N_g, -1)  # (2, B, N_g, K) — Sinkhorn targets
        p_sg = F.softmax(sg_scores.view(2, B, N_g, -1) / self.tau_s, dim=-1)  # (2, B, N_g, K)
        p_sl = F.softmax(sl_scores.view(n_local, B, N_l, -1) / self.tau_s, dim=-1)  # (n_local, B, N_l, K)

        # --- L_A: loop over images and view pairs ---
        total_la = torch.tensor(0.0, device=device)
        n_valid_pairs = 0

        for b in range(B):
            for t_v in range(2):  # two teacher global views
                t_params = crop_params_batch[b][t_v]

                for s_v in range(2 + n_local):  # student: 2 global + n_local local
                    if s_v < 2:
                        s_params = crop_params_batch[b][s_v]
                        p_s_patch = p_sg[s_v, b]  # (N_g, K)
                    else:
                        j = s_v - 2
                        s_params = crop_params_batch[b][2 + j]
                        p_s_patch = p_sl[j, b]  # (N_l, K)

                    t_idx_list, s_idx_list = compute_overlap_patch_indices(
                        t_params, s_params, patch_size=self.patch_size
                    )
                    if len(t_idx_list) == 0:
                        continue

                    t_idx = torch.tensor(t_idx_list, device=device, dtype=torch.long)
                    s_idx = torch.tensor(s_idx_list, device=device, dtype=torch.long)

                    q = q_t[t_v, b][t_idx]   # (M, K) — stop-gradient (from teacher + Sinkhorn)
                    p = p_s_patch[s_idx]      # (M, K) — has gradient via student backbone

                    # Cross-entropy: −Σ_k q_k log(p_k), mean over M patches
                    la = -(q * (p + 1e-8).log()).sum(dim=-1).mean()
                    total_la = total_la + la
                    n_valid_pairs += 1

        if n_valid_pairs > 0:
            total_la = total_la / n_valid_pairs

        return total_la
