"""Phase 1 training script: iBOT + Prototype Assignment Loss (L_A)."""

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
from chexfound.train.cxr_ssl_phase1_arch import SSLPhase1Arch
from chexfound import utils


torch.backends.cuda.matmul.allow_tf32 = True
logger = logging.getLogger("chexfound")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Phase 1 CXR SSL training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output-dir", "--output_dir", default="", type=str)
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
    # lambda_A warmup: linear ramp from 0 to lambda_A over warmup epochs
    warmup_epochs_la = cfg.prototype.get("lambda_A_warmup_epochs", 0)
    lambda_A_schedule = torch.zeros(total)
    warmup_end = warmup_epochs_la * EPOCH_LEN
    if warmup_end > 0:
        lambda_A_schedule[:warmup_end] = torch.linspace(0.0, cfg.prototype.lambda_A, warmup_end)
        lambda_A_schedule[warmup_end:] = cfg.prototype.lambda_A
    else:
        lambda_A_schedule[:] = cfg.prototype.lambda_A
    return lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule, lambda_A_schedule


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for g in optimizer.param_groups:
        g["weight_decay"] = wd * g["wd_multiplier"]
        g["lr"] = (last_layer_lr if g["is_last_layer"] else lr) * g["lr_multiplier"]


def do_test(cfg, model, iteration):
    if distributed.is_main_process():
        eval_dir = os.path.join(cfg.train.output_dir, "eval", str(iteration))
        os.makedirs(eval_dir, exist_ok=True)
        torch.save({"teacher": model.teacher.state_dict()},
                   os.path.join(eval_dir, "teacher_checkpoint.pth"))
        torch.save({"model": model.state_dict()},
                   os.path.join(eval_dir, "model_checkpoint.pth"))


@torch.no_grad()
def _log_prototype_stats(model, iteration, epoch_len):
    """Epoch-end prototype health check.

    Logs:
      - prototype_entropy: mean assignment entropy over queue vectors.
          Low → prototypes are collapsing (one dominates).
          High (= log K) → uniform, healthy.
      - dead_prototypes: number of prototypes with mean assignment < 1/K × 0.1.
          Ideally 0. Rising dead count means representation is shrinking.
      - proto_sim_mean: mean pairwise cosine similarity of prototypes.
          Should stay low (< 0.3). High → prototypes becoming redundant.
    """
    import torch.nn.functional as F
    K = model.prototype_dict.K
    proto = model.prototype_dict.prototypes.float()  # (K, d)

    # Use queue as proxy distribution (per-image means already stored)
    q_vectors = model.queue.get()  # (Q, d) fp32
    if q_vectors.shape[0] < 10:
        return  # queue not ready yet

    q_vectors = q_vectors.to(proto.device)
    scores = torch.mm(q_vectors, proto.T)           # (Q, K)
    assignment = torch.softmax(scores / 0.04, dim=-1)  # (Q, K)

    # Mean assignment per prototype → usage distribution
    mean_assign = assignment.mean(dim=0)            # (K,)

    # Entropy of mean assignment distribution (how uniformly prototypes are used)
    entropy = -(mean_assign * (mean_assign + 1e-8).log()).sum()
    max_entropy = math.log(K)

    # Dead prototypes (usage < 10% of uniform)
    dead = (mean_assign < (1.0 / K) * 0.1).sum().item()

    # Mean pairwise similarity (sample 64 pairs for speed)
    idx = torch.randperm(K, device=proto.device)[:min(64, K)]
    proto_sub = F.normalize(proto[idx], dim=-1)
    sim_matrix = torch.mm(proto_sub, proto_sub.T)
    mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=proto.device)
    proto_sim = sim_matrix[mask].mean().item()

    epoch = (iteration + 1) // epoch_len
    logger.info(
        f"[Proto stats] epoch={epoch}  "
        f"entropy={entropy:.3f}/{max_entropy:.3f} ({entropy/max_entropy*100:.1f}%)  "
        f"dead={dead}/{K}  "
        f"proto_sim={proto_sim:.3f}"
    )

    # z_R proxy: for each queue vector, compute residual after projecting onto best prototype
    # If prototypes absorb ALL signal, z_R_norm → 0. Should stay meaningfully above 0.
    with torch.no_grad():
        best_proto = proto[scores.argmax(dim=1)]          # (Q, d) — nearest prototype
        proj = (q_vectors * best_proto).sum(dim=-1, keepdim=True) * best_proto  # projection
        z_r_norm = (q_vectors - proj).norm(dim=1).mean().item()  # mean residual norm
    logger.info(
        f"[Proto stats] epoch={epoch}  "
        f"z_R_norm={z_r_norm:.4f} (0=collapsed, ~1=healthy residual)"
    )


def do_train(cfg, model, resume=False):
    model.train()
    fp16_scaler = model.fp16_scaler

    optimizer = build_optimizer(cfg, model.get_params_groups())
    lr_sched, wd_sched, mom_sched, teacher_temp_sched, last_layer_lr_sched, lambda_A_sched = build_schedulers(cfg)

    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    start_iter = checkpointer.resume_or_load('', resume=resume).get("iteration", -1) + 1

    EPOCH_LEN = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * EPOCH_LEN

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, period=500, max_iter=max_iter, max_to_keep=5
    )

    # Data setup
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=int(0.5 * (img_size // patch_size) ** 2),
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

    # Training loop
    iteration = start_iter
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)

    # PCA prototype init — fresh start only, k-means file 없을 때
    if start_iter == 0 and not cfg.prototype.get("init_path", ""):
        logger.info("Initializing prototypes via PCA from first batch...")
        _init_data = next(iter(data_loader))
        model.init_prototypes_from_pca(_init_data)
        logger.info("PCA prototype initialization complete.")

    logger.info(f"Starting Phase 1 training from iteration {start_iter}")

    for data in metric_logger.log_every(data_loader, 10, "Phase1", max_iter, start_iter):
        if data is None:
            continue
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        lr = lr_sched[iteration]
        wd = wd_sched[iteration]
        mom = mom_sched[iteration]
        teacher_temp = teacher_temp_sched[iteration]
        last_layer_lr = last_layer_lr_sched[iteration]
        lambda_A = lambda_A_sched[iteration].item()
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp, lambda_A=lambda_A)

        # All-reduce prototype gradients (not FSDP-managed)
        if distributed.is_enabled():
            world_size = distributed.get_global_size()
            for p in model.prototype_dict.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(world_size)

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

        # Project prototypes back to unit sphere (only if step was taken)
        if not _grad_nan:
            model.prototype_dict.normalize()

        # Teacher EMA update
        model.update_teacher(mom)

        # Logging — iteration always advances regardless of NaN loss
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN loss detected — batch skipped")
        else:
            losses_reduced = sum(loss_dict_reduced.values())
            metric_logger.update(lr=lr, wd=wd, mom=mom, last_layer_lr=last_layer_lr,
                                  lambda_A=lambda_A,
                                  current_batch_size=current_batch_size,
                                  total_loss=losses_reduced, **loss_dict_reduced)

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()

        # Prototype convergence monitoring — once per epoch (on rank 0 only)
        if distributed.is_main_process() and (iteration + 1) % EPOCH_LEN == 0:
            _log_prototype_stats(model, iteration, EPOCH_LEN)

        periodic_checkpointer.step(iteration)
        iteration += 1

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    model = SSLPhase1Arch(cfg).to(torch.device("cuda"))

    if cfg.MODEL.WEIGHTS:
        utils.utils.load_pretrained_weights_train(model, cfg.MODEL.WEIGHTS)

    # Optionally initialize prototypes from k-means
    if cfg.prototype.get("init_path", ""):
        model.prototype_dict.load_kmeans_init(cfg.prototype.init_path)

    model.prepare_for_distributed_training()

    logger.info("Phase 1 Model:\n{}".format(model))
    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
