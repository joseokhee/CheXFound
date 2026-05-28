"""Phase 2 training script: iBOT + L_A (0.1) + Residual Loss (L_R)."""

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
import torch
import torch.distributed as dist

from chexfound.data import SamplerType, make_data_loader, make_dataset
from chexfound.data import collate_data_and_cast_with_params, MaskingGenerator
from chexfound.data import DataAugmentationDINOWithParams
import chexfound.distributed as distributed
from chexfound.fsdp import FSDPCheckpointer
from chexfound.logging import MetricLogger
from chexfound.utils.config import setup
from chexfound.utils.utils import CosineScheduler
from chexfound.train.cxr_ssl_phase2_arch import SSLPhase2Arch


torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger("chexfound")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Phase 2 CXR SSL training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output-dir", "--output_dir", default="", type=str)
    parser.add_argument(
        "--phase1-checkpoint", "--phase1_checkpoint",
        default="",
        type=str,
        help="Path to Phase 1 model_checkpoint.pth to initialize backbone, ibot_head, prototype_dict",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options at end of command (space-separated PATH.KEY VALUE pairs)",
    )
    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    EPOCH_LEN = cfg.train.OFFICIAL_EPOCH_LENGTH
    total = cfg.optim.epochs * EPOCH_LEN
    warmup = cfg.optim.warmup_epochs * EPOCH_LEN
    lr_schedule = CosineScheduler(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=total,
        warmup_iters=warmup,
        start_warmup_value=0,
    )
    wd_schedule = CosineScheduler(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=total,
    )
    momentum_schedule = CosineScheduler(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=total,
    )
    teacher_temp_schedule = CosineScheduler(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * EPOCH_LEN,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * EPOCH_LEN,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )
    last_layer_lr_schedule = CosineScheduler(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=total,
        warmup_iters=warmup,
        start_warmup_value=0,
    )
    last_layer_lr_schedule.schedule[: cfg.optim["freeze_last_layer_epochs"] * EPOCH_LEN] = 0
    return lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for g in optimizer.param_groups:
        g["weight_decay"] = wd * g["wd_multiplier"]
        g["lr"] = (last_layer_lr if g["is_last_layer"] else lr) * g["lr_multiplier"]


def load_phase1_checkpoint(model, ckpt_path):
    """Load Phase 1 model_checkpoint.pth into Phase 2 model.

    Loads backbone, ibot_head (student+teacher), and prototype_dict.
    Ignores missing/unexpected keys silently.
    """
    logger.info(f"Loading Phase 1 checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    # model_checkpoint.pth is saved as {"model": model.state_dict()}
    if "model" in state_dict:
        state_dict = state_dict["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Phase 1 checkpoint loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        logger.info(f"Missing: {missing[:10]}")


def do_test(cfg, model, iteration):
    if distributed.is_main_process():
        eval_dir = os.path.join(cfg.train.output_dir, "eval", str(iteration))
        os.makedirs(eval_dir, exist_ok=True)
        torch.save({"teacher": model.teacher.state_dict()},
                   os.path.join(eval_dir, "teacher_checkpoint.pth"))
        torch.save({"model": model.state_dict()},
                   os.path.join(eval_dir, "model_checkpoint.pth"))


def do_train(cfg, model, resume=False):
    model.train()
    fp16_scaler = model.fp16_scaler

    # Prototype dict is frozen in Phase 2 — exclude from optimizer
    # get_params_groups() includes prototype params; we filter them out if frozen
    params_groups = [
        g for g in model.get_params_groups()
        if any(p.requires_grad for p in g["params"])
    ]
    optimizer = build_optimizer(cfg, params_groups)
    lr_sched, wd_sched, mom_sched, teacher_temp_sched, last_layer_lr_sched = build_schedulers(cfg)

    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    start_iter = checkpointer.resume_or_load('', resume=resume).get("iteration", -1) + 1

    EPOCH_LEN = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * EPOCH_LEN

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, period=3 * EPOCH_LEN, max_iter=max_iter, max_to_keep=3
    )

    # Data setup
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=int(cfg.ibot.mask_ratio_min_max[1] * (img_size // patch_size) ** 2),
    )

    data_transform = DataAugmentationDINOWithParams(
        global_crops_scale=cfg.crops.global_crops_scale,
        local_crops_scale=cfg.crops.local_crops_scale,
        local_crops_number=cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )

    collate_fn = partial(
        collate_data_and_cast_with_params,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=torch.half,
    )

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,
        sampler_type=SamplerType.SHARDED_INFINITE,
        sampler_advance=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    iteration = start_iter
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)

    logger.info(f"Starting Phase 2 training from iteration {start_iter}")

    for data in metric_logger.log_every(data_loader, 10, "Phase2", max_iter, start_iter):
        if data is None:
            continue
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        current_epoch = iteration // EPOCH_LEN

        lr = lr_sched[iteration]
        wd = wd_sched[iteration]
        mom = mom_sched[iteration]
        teacher_temp = teacher_temp_sched[iteration]
        last_layer_lr = last_layer_lr_sched[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp, current_epoch=current_epoch)

        # Gradient clipping and optimizer step
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            # NaN gradient check — must be before step to prevent parameter contamination
            _grad_nan = any(
                (p.grad is not None and not torch.isfinite(p.grad).all())
                for pg in optimizer.param_groups for p in pg["params"]
            )
            if _grad_nan:
                logger.info("NaN gradient detected — skipping optimizer.step()")
                optimizer.zero_grad(set_to_none=True)
                fp16_scaler.update()
            else:
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            # NaN gradient check
            _grad_nan = any(
                (p.grad is not None and not torch.isfinite(p.grad).all())
                for pg in optimizer.param_groups for p in pg["params"]
            )
            if _grad_nan:
                logger.info("NaN gradient detected — skipping optimizer.step()")
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()

        # Teacher EMA update
        model.update_teacher(mom)

        # Logging
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN loss detected — batch skipped")
            continue

        losses_reduced = sum(loss_dict_reduced.values())
        metric_logger.update(lr=lr, wd=wd, mom=mom, last_layer_lr=last_layer_lr,
                              current_epoch=current_epoch,
                              current_batch_size=current_batch_size,
                              total_loss=losses_reduced, **loss_dict_reduced)

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()

        periodic_checkpointer.step(iteration)
        iteration += 1

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    model = SSLPhase2Arch(cfg).to(torch.device("cuda"))

    # Load Phase 1 checkpoint (backbone + ibot_head + prototype_dict)
    if args.phase1_checkpoint:
        load_phase1_checkpoint(model, args.phase1_checkpoint)
    elif cfg.MODEL.WEIGHTS:
        from chexfound import utils
        utils.utils.load_pretrained_weights_train(model, cfg.MODEL.WEIGHTS)

    # Freeze prototype dictionary
    model.freeze_prototype_dict()

    model.prepare_for_distributed_training()

    logger.info("Phase 2 Model:\n{}".format(model))
    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
