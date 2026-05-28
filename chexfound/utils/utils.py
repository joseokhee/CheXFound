# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import random
import subprocess
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn


logger = logging.getLogger("chexfound")


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def load_pretrained_weights_train(model, pretrained_weights):
    if os.path.basename(pretrained_weights).startswith('dinov2_'):  # pure state_dict (no 'model' wrapper)
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        state_dict = {"student.backbone."+k: v for k, v in state_dict.items()}
        state_dict.pop("student.backbone.pos_embed", None)
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")['model']
        state_dict.pop("teacher.backbone.pos_embed", None)
        state_dict.pop("student.backbone.pos_embed", None)

    # Remap flat block keys (blocks.N.*) to block_chunks format (blocks.C.N.*)
    # This handles checkpoints saved without block_chunks (e.g. block_chunks=1)
    # For block_chunks=4 with 24 blocks: chunk_size=6, blocks.N -> blocks.N//6.N
    model_state = model.state_dict()
    # Detect if remapping is needed: check if any flat block key exists but chunked does not
    import re
    remapped = {}
    for k, v in state_dict.items():
        m = re.match(r'^(.*\.backbone\.blocks\.)(\d+)(\..+)$', k)
        if m:
            prefix, block_idx_str, suffix = m.group(1), m.group(2), m.group(3)
            block_idx = int(block_idx_str)
            flat_key = k
            # Check if model uses chunked format
            if flat_key not in model_state:
                # Try to find the chunk this block belongs to
                # Enumerate model keys to find the matching chunk index
                found = False
                for chunk_idx in range(10):
                    chunked_key = f"{prefix}{chunk_idx}.{block_idx}{suffix}"
                    if chunked_key in model_state:
                        remapped[chunked_key] = v
                        found = True
                        break
                if not found:
                    remapped[k] = v  # keep original if no match
            else:
                remapped[k] = v
        else:
            remapped[k] = v
    state_dict = remapped

    # Remove keys with shape mismatch to allow safe partial loading
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape != v.shape:
            skipped.append(f"{k}: ckpt {tuple(v.shape)} vs model {tuple(model_state[k].shape)}")
        else:
            filtered[k] = v
    if skipped:
        logger.info("Skipping {} keys due to shape mismatch:\n  {}".format(
            len(skipped), "\n  ".join(skipped)))

    msg = model.load_state_dict(filtered, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
