#!/bin/bash
# Phase 1 CXR SSL Training: iBOT + Prototype Assignment Loss (L_A)
# ViT-L/14, 518x518, 3 GPUs

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=3600        # 1 hour — prevent NCCL timeout on slow first batch
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
CSV_PATH="/data4/workspaces/shjo/CXR/datasets/CXR_ALL/dataset.csv"
OUTPUT_DIR="/data4/workspaces/shjo/CXR/Pretrain/outputs/phase1_prototype_v3"

mkdir -p "${OUTPUT_DIR}"

conda run --no-capture-output -n CheXFound \
  torchrun --nproc_per_node=3 --master_port=29504 \
  chexfound/train/train_phase1.py \
  --config-file chexfound/configs/train/vitl14_phase1.yaml \
  --output-dir "${OUTPUT_DIR}" \
  train.dataset_path="CXRDatabaseCSV:root=${CSV_PATH}" \
  train.batch_size_per_gpu=18 \
  train.num_workers=2 \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
