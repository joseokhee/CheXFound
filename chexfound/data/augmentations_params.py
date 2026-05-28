import logging
import random

import numpy as np

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from .transforms import GaussianBlur, make_normalize_transform


logger = logging.getLogger("chexfound")


def _geometric_augment_with_params(image, output_size, scale, ratio=(3 / 4, 4 / 3)):
    """RandomResizedCrop + RandomHorizontalFlip, capturing all spatial parameters."""
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
    image = TF.resized_crop(image, i, j, h, w, size=(output_size, output_size),
                            interpolation=InterpolationMode.BICUBIC)
    flipped = random.random() < 0.5
    if flipped:
        image = TF.hflip(image)
    params = {
        "top": i, "left": j, "height": h, "width": w,
        "output_size": output_size, "flipped": flipped,
    }
    return image, params


class DataAugmentationDINOWithParams:
    """Drop-in replacement for DataAugmentationDINO that also returns crop parameters.

    The returned dict has the same keys as DataAugmentationDINO plus:
      "crop_params": list of (2 + local_crops_number) dicts, one per view.
        Each dict: {top, left, height, width, output_size, flipped, view, view_idx}
        Views are ordered: [global_0, global_1, local_0, ..., local_{n-1}]
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using DataAugmentationDINOWithParams:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        color_jittering = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            make_normalize_transform(),
        ])

        self.global_transfo1 = transforms.Compose([color_jittering, GaussianBlur(p=1.0), self.normalize])
        self.global_transfo2 = transforms.Compose([
            color_jittering,
            GaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            self.normalize,
        ])
        self.local_transfo = transforms.Compose([color_jittering, GaussianBlur(p=0.5), self.normalize])

    def __call__(self, image):
        # --- global crop 0 ---
        im1_base, params_g0 = _geometric_augment_with_params(
            image, self.global_crops_size, self.global_crops_scale
        )
        global_crop_1 = self.global_transfo1(im1_base)
        params_g0.update({"view": "global", "view_idx": 0})

        # --- global crop 1 ---
        im2_base, params_g1 = _geometric_augment_with_params(
            image, self.global_crops_size, self.global_crops_scale
        )
        global_crop_2 = self.global_transfo2(im2_base)
        params_g1.update({"view": "global", "view_idx": 1})

        # --- local crops ---
        local_crops = []
        local_params = []
        for k in range(self.local_crops_number):
            im_local, params_lk = _geometric_augment_with_params(
                image, self.local_crops_size, self.local_crops_scale
            )
            local_crops.append(self.local_transfo(im_local))
            params_lk.update({"view": "local", "view_idx": k})
            local_params.append(params_lk)

        all_crop_params = [params_g0, params_g1] + local_params

        return {
            "global_crops": [global_crop_1, global_crop_2],
            "global_crops_teacher": [global_crop_1, global_crop_2],
            "local_crops": local_crops,
            "crop_params": all_crop_params,
            "offsets": (),
        }


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------

def compute_overlap_patch_indices(crop_a, crop_b, patch_size=14):
    """Find matching patch index pairs between crop_a and crop_b.

    Uses center-point matching vectorized over all grid_a patches at once with
    numpy, avoiding Python-level nested loops.

    Args:
        crop_a: dict with keys top, left, height, width, output_size, flipped
        crop_b: dict with keys top, left, height, width, output_size, flipped
        patch_size: ViT patch size in pixels

    Returns:
        (a_indices, b_indices): flat patch indices (int lists).
        Empty lists when there is no spatial overlap.
    """
    grid_a = crop_a["output_size"] // patch_size
    grid_b = crop_b["output_size"] // patch_size

    t_a, l_a = crop_a["top"], crop_a["left"]
    h_a, w_a = crop_a["height"], crop_a["width"]
    b_a, r_a = t_a + h_a, l_a + w_a

    t_b, l_b = crop_b["top"], crop_b["left"]
    h_b, w_b = crop_b["height"], crop_b["width"]
    b_b, r_b = t_b + h_b, l_b + w_b

    # Quick rejection
    if t_a >= b_b or t_b >= b_a or l_a >= r_b or l_b >= r_a:
        return [], []

    # Intersection region in original image coordinates
    inter_t = max(t_a, t_b)
    inter_l = max(l_a, l_b)
    inter_b_ = min(b_a, b_b)
    inter_r = min(r_a, r_b)

    # Build row/col grids for all patches in crop_a at once
    rows_a = np.arange(grid_a, dtype=np.float32)   # (grid_a,)
    cols_a = np.arange(grid_a, dtype=np.float32)   # (grid_a,)

    # Center coordinates in original image space
    center_y = t_a + (rows_a + 0.5) / grid_a * h_a  # (grid_a,)
    if crop_a["flipped"]:
        orig_col_a = (grid_a - 1) - cols_a
    else:
        orig_col_a = cols_a
    center_x = l_a + (orig_col_a + 0.5) / grid_a * w_a  # (grid_a,)

    # Valid row mask (same for all columns)
    valid_row = (center_y >= inter_t) & (center_y < inter_b_)  # (grid_a,)
    valid_col = (center_x >= inter_l) & (center_x < inter_r)   # (grid_a,)

    # Build 2-D masks via outer product
    valid_mask = np.outer(valid_row, valid_col)  # (grid_a, grid_a)

    if not valid_mask.any():
        return [], []

    # Compute crop_b row/col for every patch in crop_a
    row_b = (center_y - t_b) / h_b * grid_b   # (grid_a,)
    row_b_idx = row_b.astype(np.int32)          # (grid_a,)

    col_b_orig = (center_x - l_b) / w_b * grid_b  # (grid_a,)
    col_b_orig_idx = col_b_orig.astype(np.int32)    # (grid_a,)
    if crop_b["flipped"]:
        col_b_idx = (grid_b - 1) - col_b_orig_idx
    else:
        col_b_idx = col_b_orig_idx

    # Also mask out out-of-bound indices in crop_b
    valid_rb = (row_b_idx >= 0) & (row_b_idx < grid_b)   # (grid_a,)
    valid_cb = (col_b_orig_idx >= 0) & (col_b_orig_idx < grid_b)  # (grid_a,)
    valid_mask &= np.outer(valid_rb, valid_cb)

    if not valid_mask.any():
        return [], []

    # Gather indices where valid
    row_a_vec, col_a_vec = np.where(valid_mask)  # flat index vectors

    row_b_vec = row_b_idx[row_a_vec]
    col_b_vec = col_b_idx[col_a_vec]

    a_indices = (row_a_vec * grid_a + col_a_vec).tolist()
    b_indices = (row_b_vec * grid_b + col_b_vec).tolist()

    return a_indices, b_indices
