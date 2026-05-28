#!/bin/bash
# CXRDatabaseCSV 기반 학습 스크립트
# 원본 파일 복사 없이 dataset.csv에서 직접 로딩

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2
export NCCL_P2P_DISABLE=1       # GPU간 P2P hang 방지

CSV_PATH="/data4/workspaces/shjo/CXR/datasets/CXR_ALL/dataset.csv"
OUTPUT_DIR="/data4/workspaces/shjo/CXR/Pretrain/outputs/chexfound_csv_v2"

mkdir -p "${OUTPUT_DIR}"

conda run --no-capture-output -n CheXFound \
  torchrun --nproc_per_node=3 --master_port=29503 \
  chexfound/train/train.py \
  --config-file chexfound/configs/train/vitl16_ibot333_highres512.yaml \
  --output-dir "${OUTPUT_DIR}" \
  train.dataset_path="CXRDatabaseCSV:root=${CSV_PATH}" \
  train.batch_size_per_gpu=24 \
  train.num_workers=4 \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
