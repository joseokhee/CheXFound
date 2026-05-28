"""Smoke test for Phase 1 and Phase 2 architectures.

Run with:
    CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=29510 \
        scripts/smoke_test.py [--phase2]
"""

import argparse
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import chexfound.distributed as distributed


# ── helpers ────────────────────────────────────────────────────────────────

def make_fake_crop_params(B, n_local, orig_h=1000, orig_w=1000):
    """Generate plausible crop_params for B images, 2 global + n_local local."""
    batch = []
    for _ in range(B):
        params = []
        # Two global crops (large, overlapping)
        for v in range(2):
            t = torch.randint(0, orig_h // 4, (1,)).item()
            l = torch.randint(0, orig_w // 4, (1,)).item()
            h = torch.randint(orig_h // 2, orig_h - t, (1,)).item()
            w = torch.randint(orig_w // 2, orig_w - l, (1,)).item()
            params.append({
                "top": t, "left": l, "height": h, "width": w,
                "output_size": 518,
                "flipped": bool(torch.randint(0, 2, (1,)).item()),
                "view": "global", "view_idx": v,
            })
        # Local crops (small)
        for k in range(n_local):
            t = torch.randint(0, orig_h * 3 // 4, (1,)).item()
            l = torch.randint(0, orig_w * 3 // 4, (1,)).item()
            h = torch.randint(orig_h // 6, orig_h // 3, (1,)).item()
            w = torch.randint(orig_w // 6, orig_w // 3, (1,)).item()
            h = min(h, orig_h - t)
            w = min(w, orig_w - l)
            params.append({
                "top": t, "left": l, "height": max(h, 50), "width": max(w, 50),
                "output_size": 140,
                "flipped": bool(torch.randint(0, 2, (1,)).item()),
                "view": "local", "view_idx": k,
            })
        batch.append(params)
    return batch


def make_fake_batch(B, n_local, n_g_patches, mask_ratio=0.3, device="cuda"):
    """Build a fake data dict matching collate_data_and_cast_with_params output."""
    dtype = torch.half

    global_crops = torch.randn(2 * B, 3, 518, 518, dtype=dtype, device=device)
    local_crops = torch.randn(n_local * B, 3, 140, 140, dtype=dtype, device=device)

    # Create masks: ~mask_ratio of patches masked per global crop
    n_masked_per = int(n_g_patches * mask_ratio)
    masks_list = []
    for _ in range(2 * B):
        m = torch.zeros(n_g_patches, dtype=torch.bool)
        idx = torch.randperm(n_g_patches)[:n_masked_per]
        m[idx] = True
        masks_list.append(m)
    masks = torch.stack(masks_list, dim=0).to(device)  # (2B, N_g)

    mask_indices_list = masks.flatten().nonzero(as_tuple=False).squeeze(1)
    n_masked = mask_indices_list.shape[0]
    upperbound = n_masked + 64  # small slack

    # masks_weight: uniform (1 / n_masked_per) per image
    masks_weight = (1.0 / masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(masks)[masks]

    crop_params = make_fake_crop_params(B, n_local)

    return {
        "collated_global_crops": global_crops,
        "collated_local_crops": local_crops,
        "collated_masks": masks,
        "mask_indices_list": mask_indices_list,
        "n_masked_patches": torch.full((1,), n_masked, dtype=torch.long, device=device),
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "crop_params": crop_params,
    }


def check_losses(loss_dict, tag=""):
    ok = True
    for k, v in loss_dict.items():
        val = v.item() if torch.is_tensor(v) else v
        nan = math.isnan(val) or math.isinf(val)
        status = "NaN/Inf !!!" if nan else "OK"
        print(f"  {tag}{k}: {val:.4f}  [{status}]")
        if nan:
            ok = False
    return ok


# ── Phase 1 test ───────────────────────────────────────────────────────────

def test_phase1(cfg, device):
    print("\n" + "=" * 60)
    print("PHASE 1 SMOKE TEST")
    print("=" * 60)

    from chexfound.train.cxr_ssl_phase1_arch import SSLPhase1Arch

    print("Building SSLPhase1Arch...")
    model = SSLPhase1Arch(cfg).to(device)
    model.prepare_for_distributed_training()
    model.train()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Prototype params: {sum(p.numel() for p in model.prototype_dict.parameters()):,}")

    optimizer = torch.optim.AdamW(model.get_params_groups(), lr=1e-4)

    B = 2
    n_local = cfg.crops.local_crops_number
    n_g = (cfg.crops.global_crops_size // cfg.student.patch_size) ** 2

    teacher_temp = 0.04
    n_iters = 3

    print(f"\nRunning {n_iters} iterations (B={B}, N_g={n_g}, n_local={n_local})...")
    all_ok = True

    for i in range(n_iters):
        data = make_fake_batch(B, n_local, n_g, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)
        optimizer.step()
        model.prototype_dict.normalize()
        model.update_teacher(m=0.9995)

        print(f"\n  Iter {i+1}:")
        ok = check_losses(loss_dict, tag="    ")
        all_ok = all_ok and ok

    # Test queue state
    print(f"\n  Queue fill_count: {model.queue.fill_count:,} / {model.queue.queue_size:,}")
    print(f"  Queue is_ready: {model.queue.is_ready()}")

    # Check prototype norms (should be ~1.0)
    proto_norms = model.prototype_dict.prototypes.data.norm(dim=1)
    print(f"  Prototype norms: min={proto_norms.min():.4f}, max={proto_norms.max():.4f}, mean={proto_norms.mean():.4f}")

    print(f"\nPhase 1: {'PASSED' if all_ok else 'FAILED'}")
    return model, all_ok


# ── Phase 2 test ───────────────────────────────────────────────────────────

def test_phase2(cfg_p2, device, phase1_model=None):
    print("\n" + "=" * 60)
    print("PHASE 2 SMOKE TEST")
    print("=" * 60)

    from chexfound.train.cxr_ssl_phase2_arch import SSLPhase2Arch

    print("Building SSLPhase2Arch...")
    model = SSLPhase2Arch(cfg_p2).to(device)

    # Simulate loading Phase 1 checkpoint by copying prototype_dict
    if phase1_model is not None:
        with torch.no_grad():
            model.prototype_dict.prototypes.copy_(
                phase1_model.prototype_dict.prototypes
            )
        print("  Copied Phase 1 prototype_dict")

    model.freeze_prototype_dict()
    model.prepare_for_distributed_training()
    model.train()

    # Filter out frozen prototype params from optimizer
    params_groups = [g for g in model.get_params_groups() if any(p.requires_grad for p in g["params"])]
    optimizer = torch.optim.AdamW(params_groups, lr=5e-5)

    B = 2
    n_local = cfg_p2.crops.local_crops_number
    n_g = (cfg_p2.crops.global_crops_size // cfg_p2.student.patch_size) ** 2

    teacher_temp = 0.04
    n_iters = 3

    print(f"\nRunning {n_iters} iterations (B={B}, N_g={n_g}, mask=70%)...")
    all_ok = True

    for i in range(n_iters):
        # Phase 2 uses fixed 70% masking
        data = make_fake_batch(B, n_local, n_g, mask_ratio=0.7, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp, current_epoch=25)
        optimizer.step()
        model.update_teacher(m=0.9995)

        print(f"\n  Iter {i+1} (epoch=25, dynamic weighting ON):")
        ok = check_losses(loss_dict, tag="    ")
        all_ok = all_ok and ok

    # Also test warmup epoch (no dynamic weighting)
    print(f"\n  Iter with epoch=5 (dynamic weighting OFF):")
    data = make_fake_batch(B, n_local, n_g, mask_ratio=0.7, device=device)
    optimizer.zero_grad(set_to_none=True)
    loss_dict = model.forward_backward(data, teacher_temp=teacher_temp, current_epoch=5)
    ok = check_losses(loss_dict, tag="    ")
    all_ok = all_ok and ok

    print(f"\nPhase 2: {'PASSED' if all_ok else 'FAILED'}")
    return all_ok


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2", action="store_true", help="Also test Phase 2")
    args = parser.parse_args()

    # Initialize distributed (required by FSDP / sinkhorn)
    if not distributed.is_enabled():
        distributed.enable(overwrite=True)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Distributed world_size: {distributed.get_global_size()}")

    base_cfg = OmegaConf.load("chexfound/configs/ssl_default_config.yaml")
    p1_cfg = OmegaConf.merge(base_cfg, OmegaConf.load("chexfound/configs/train/vitl14_phase1.yaml"))
    p1_cfg.train.output_dir = "/tmp/smoke_phase1"
    p1_cfg.optim.lr = p1_cfg.optim.base_lr  # skip scaling rule

    os.makedirs("/tmp/smoke_phase1", exist_ok=True)
    os.makedirs("/tmp/smoke_phase2", exist_ok=True)

    phase1_model, p1_ok = test_phase1(p1_cfg, device)

    if args.phase2:
        p2_cfg = OmegaConf.merge(base_cfg, OmegaConf.load("chexfound/configs/train/vitl14_phase2.yaml"))
        p2_cfg.train.output_dir = "/tmp/smoke_phase2"
        p2_cfg.optim.lr = p2_cfg.optim.base_lr
        p2_ok = test_phase2(p2_cfg, device, phase1_model=phase1_model)
    else:
        p2_ok = True

    print("\n" + "=" * 60)
    print(f"FINAL: Phase1={'PASS' if p1_ok else 'FAIL'}  Phase2={'PASS' if p2_ok else 'FAIL (skipped)' if not args.phase2 else 'FAIL'}")
    print("=" * 60)

    if not (p1_ok and p2_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
