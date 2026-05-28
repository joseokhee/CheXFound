#!/bin/bash
# Phase 2 CXR SSL Training: iBOT + L_A (0.1) + Residual Loss (L_R)
# ViT-L/14, 518x518, 70% masking, 3 GPUs

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2
export NCCL_P2P_DISABLE=1

CSV_PATH="/data4/workspaces/shjo/CXR/datasets/CXR_ALL/dataset.csv"
PHASE1_OUTPUT="/data4/workspaces/shjo/CXR/Pretrain/outputs/phase1_prototype"
OUTPUT_DIR="/data4/workspaces/shjo/CXR/Pretrain/outputs/phase2_residual"

# Set path to the Phase 1 checkpoint (adjust iteration number as needed)
PHASE1_CKPT="${PHASE1_OUTPUT}/eval/training_XXXXX/model_checkpoint.pth"

mkdir -p "${OUTPUT_DIR}"

conda run --no-capture-output -n CheXFound \
  torchrun --nproc_per_node=3 --master_port=29505 \
  chexfound/train/train_phase2.py \
  --config-file chexfound/configs/train/vitl14_phase2.yaml \
  --output-dir "${OUTPUT_DIR}" \
  --phase1-checkpoint "${PHASE1_CKPT}" \
  train.dataset_path="CXRDatabaseCSV:root=${CSV_PATH}" \
  train.batch_size_per_gpu=14 \
  train.num_workers=4 \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
