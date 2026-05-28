"""K-means prototype initialization using DINOv2 ViT-L/14 patch embeddings.

Extracts patch embeddings from the backbone (no ibot_head), runs MiniBatchKMeans
with K=512 clusters, L2-normalizes the centres, and saves to a .npy file.

Usage:
    python scripts/init_prototypes.py \
        --weights /path/to/dinov2_vitl14.pth \
        --csv /path/to/dataset.csv \
        --output-dir /path/to/output \
        --n-samples 50000 \
        --K 512
"""

import argparse
import logging
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chexfound.models import build_model_from_cfg
from chexfound.data import make_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("init_prototypes")


def get_args():
    parser = argparse.ArgumentParser("K-means prototype initialization")
    parser.add_argument("--weights", required=True, help="Path to dinov2_vitl14.pth")
    parser.add_argument("--csv", required=True, help="Path to dataset.csv")
    parser.add_argument("--output-dir", default=".", help="Directory to save prototype_init.npy")
    parser.add_argument("--K", type=int, default=512, help="Number of prototypes")
    parser.add_argument("--n-samples", type=int, default=50000,
                        help="Approximate number of patch vectors to collect")
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_backbone(weights_path, img_size, patch_size):
    """Build ViT-L/14 backbone and load weights."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "student": {
            "arch": "vit_large",
            "patch_size": patch_size,
            "drop_path_rate": 0.0,
            "ffn_layer": "swiglufused",
            "block_chunks": 4,
            "num_register_tokens": 4,
            "pretrained_weights": "",
            "layerscale": 1e-5,
            "drop_path_uniform": True,
            "qkv_bias": True,
            "proj_bias": True,
            "ffn_bias": True,
            "interpolate_antialias": False,
            "interpolate_offset": 0.1,
        },
        "crops": {"global_crops_size": img_size},
    })

    from chexfound.models.vision_transformer import vit_large
    backbone = vit_large(
        patch_size=patch_size,
        img_size=img_size,
        ffn_layer="swiglufused",
        block_chunks=4,
        num_register_tokens=4,
        init_values=1e-5,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )

    # Load weights
    import re
    state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
        # Strip student.backbone. prefix if present
        state_dict = {k.replace("student.backbone.", ""): v
                      for k, v in state_dict.items()
                      if k.startswith("student.backbone.")}
    # Handle block_chunks remapping
    model_sd = backbone.state_dict()
    remapped = {}
    for k, v in state_dict.items():
        m = re.match(r'^(blocks\.)(\d+)(\..+)$', k)
        if m and k not in model_sd:
            prefix, idx_str, suffix = m.group(1), m.group(2), m.group(3)
            idx = int(idx_str)
            for chunk_i in range(10):
                ck = f"{prefix}{chunk_i}.{idx}{suffix}"
                if ck in model_sd:
                    remapped[ck] = v
                    break
            else:
                remapped[k] = v
        else:
            remapped[k] = v
    state_dict = remapped

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    logger.info(f"Backbone loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    backbone.eval()
    return backbone


def build_transform(img_size):
    from torchvision.transforms import InterpolationMode
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


@torch.no_grad()
def collect_patch_embeddings(backbone, data_loader, n_samples, device):
    """Forward pass through backbone and collect all patch embeddings."""
    all_patches = []
    total_collected = 0

    backbone = backbone.to(device)

    for images, _ in data_loader:
        images = images.to(device)
        out = backbone(images, is_training=True)
        patches = out["x_norm_patchtokens"]  # (B, N, d)
        patches = F.normalize(patches.flatten(0, 1), dim=-1)  # (B*N, d)
        all_patches.append(patches.cpu().float())
        total_collected += patches.shape[0]
        logger.info(f"Collected {total_collected:,} / {n_samples:,} patch vectors")
        if total_collected >= n_samples:
            break

    return torch.cat(all_patches, dim=0)[:n_samples].numpy()


def run_kmeans(embeddings, K, seed):
    from sklearn.cluster import MiniBatchKMeans
    logger.info(f"Running MiniBatchKMeans with K={K} on {len(embeddings):,} vectors...")
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        random_state=seed,
        batch_size=4096,
        max_iter=300,
        n_init=3,
        verbose=1,
    )
    kmeans.fit(embeddings)
    centres = kmeans.cluster_centers_.astype(np.float32)  # (K, d)
    # L2 normalize
    norms = np.linalg.norm(centres, axis=1, keepdims=True).clip(min=1e-8)
    centres = centres / norms
    return centres


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    backbone = build_backbone(args.weights, args.img_size, args.patch_size)
    transform = build_transform(args.img_size)

    dataset = make_dataset(
        dataset_str=f"CXRDatabaseCSV:root={args.csv}",
        transform=transform,
        target_transform=lambda _: 0,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    embeddings = collect_patch_embeddings(backbone, data_loader, args.n_samples, device)
    logger.info(f"Collected embeddings shape: {embeddings.shape}")

    centres = run_kmeans(embeddings, args.K, args.seed)
    logger.info(f"K-means done. Centres shape: {centres.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "prototype_init.npy")
    np.save(out_path, centres)
    logger.info(f"Saved prototype init to {out_path}")


if __name__ == "__main__":
    main()
