"""
CXRDatabaseCSV
==============
CSV 파일(path, source, fmt)을 읽어서 npy/jpg/png를 직접 로딩하는 Dataset.
파일 복사/변환 없이 원본 경로에서 바로 읽습니다.

CSV 포맷:
    path,source,fmt
    /data3/.../image.npy,chexpert,npy
    /data3/.../image.jpg,chexpert,jpg
    ...

사용 예:
    dataset = CXRDatabaseCSV(
        csv_path="/data4/workspaces/shjo/CXR/datasets/CXR_ALL/dataset.csv",
        transform=...,
    )
"""

import csv
import io
import logging
import os
import pickle
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset

logger = logging.getLogger(__name__)


class CXRDatabaseCSV(VisionDataset):
    """CSV 기반 CXR Dataset. npy/jpg/png 모두 직접 로딩.
    
    첫 로드 시 CSV를 파싱해 pickle 캐시를 만들고, 이후에는 캐시를 사용합니다.
    """

    def __init__(
        self,
        csv_path: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root=str(Path(csv_path).parent),
                         transforms=transforms,
                         transform=transform,
                         target_transform=target_transform)
        self.csv_path = csv_path
        self.samples = []   # list of (path, source, fmt)
        self.sources = []
        self._source_to_idx = {}
        self._load_csv()

    def _load_csv(self):
        cache_path = self.csv_path + ".cache.pkl"

        # 캐시가 있고 CSV보다 최신이면 캐시 사용
        if (os.path.exists(cache_path) and
                os.path.getmtime(cache_path) > os.path.getmtime(self.csv_path)):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self.samples = data["samples"]
            self.sources = data["sources"]
            self._source_to_idx = data["source_to_idx"]
            return

        # CSV 파싱
        source_set = {}
        samples = []
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row["path"]
                source = row["source"]
                fmt = row["fmt"]
                if source not in source_set:
                    source_set[source] = len(source_set)
                samples.append((path, source, fmt))

        self.samples = samples
        self.sources = list(source_set.keys())
        self._source_to_idx = source_set

        # 캐시 저장 (rank 0 경쟁 방지를 위해 tmp 파일로 먼저 쓰고 rename)
        tmp_path = cache_path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump({
                    "samples": self.samples,
                    "sources": self.sources,
                    "source_to_idx": self._source_to_idx,
                }, f)
            os.replace(tmp_path, cache_path)
        except Exception:
            pass  # 캐시 저장 실패해도 동작에는 무관

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str, fmt: str) -> Image.Image:
        if fmt == "npy":
            arr = np.load(path, allow_pickle=False)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            if arr.dtype != np.uint8:
                lo, hi = arr.min(), arr.max()
                if hi > lo:
                    arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            return Image.fromarray(arr, mode="L").convert("RGB")
        else:
            img = Image.open(path)
            arr = np.array(img).astype(np.float32)
            # 16-bit PNG (uint16) 등 non-uint8 포맷 정규화
            if arr.max() > 255 or arr.dtype != np.uint8:
                lo, hi = arr.min(), arr.max()
                if hi > lo:
                    arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
                else:
                    arr = np.zeros_like(arr, dtype=np.uint8)
            else:
                arr = arr.astype(np.uint8)
            # grayscale이면 (H,W) → L mode
            if arr.ndim == 2:
                return Image.fromarray(arr, mode="L").convert("RGB")
            return Image.fromarray(arr).convert("RGB")

    def get_image_data(self, index: int) -> bytes:
        """PIL Image → raw bytes (PNG) — CheXFound decoder 호환"""
        path, source, fmt = self.samples[index]
        img = self._load_image(path, fmt)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def get_target(self, index: int) -> int:
        _, source, _ = self.samples[index]
        return self._source_to_idx[source]

    def __getitem__(self, index: int) -> Tuple:
        for attempt in range(10):
            idx = (index + attempt) % len(self.samples)
            path, source, fmt = self.samples[idx]
            try:
                img = self._load_image(path, fmt)
            except Exception as e:
                logger.warning(f"[CXRDataset] skip bad file (attempt {attempt+1}): {path} — {e}")
                continue
            target = self._source_to_idx[source]
            try:
                if self.transforms is not None:
                    img, target = self.transforms(img, target)
                elif self.transform is not None:
                    img = self.transform(img)
            except Exception as e:
                logger.warning(f"[CXRDataset] transform failed (attempt {attempt+1}): {path} — {e}")
                continue
            return img, target
        logger.warning(f"[CXRDataset] 10 consecutive bad files at index {index}, skipping")
        return None

