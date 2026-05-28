# 구현 보조 문서 — CheXFound 코드베이스 기반 CXR SSL

본 문서는 `CXR_SSL_Architecture_Final.md`의 구현 시 필요한 세부 디테일을 보충한다. [CheXFound](https://github.com/RPIDIAL/CheXFound) 코드베이스를 fork하여 수정하는 것을 전제로 한다.

## 0. CheXFound 코드베이스 구조 및 수정 지점

### 0.1 코드베이스 개요

CheXFound은 DINOv2를 CXR 도메인에 맞게 수정한 코드베이스이다.

- **학습 진입점:** `chexfound/train/train.py`
- **설정 파일:** `chexfound/configs/train/vitl16_ibot333_highres512.yaml`
- **모델:** ViT-L/16, 이미지 해상도 512×512
- **패치 수:** 512/16 = 32 → 32×32 = 1024 패치 per image (224가 아닌 512 기준)
- **학습 방식:** torchrun 기반 multi-GPU 분산 학습 (8 GPU)

### 0.2 주요 수정 지점

| 수정 대상 | 위치 (예상) | 내용 |
|---|---|---|
| CLS self-distillation 제거 | `train.py` 또는 loss 계산 부분 | DINO CLS loss 비활성화 |
| Prototype dictionary 추가 | 모델 정의 부분 | nn.Parameter (512×1024) 추가, Sinkhorn 구현 |
| L_A loss 추가 | loss 계산 부분 | prototype assignment cross-entropy |
| Queue 추가 | 학습 루프 | Teacher z_patch 저장용 FIFO queue |
| Phase 2 학습 코드 | train.py 또는 별도 스크립트 | residual projection, L_R, dynamic weighting |
| config yaml | configs/train/ | Phase 1, Phase 2 각각의 설정 파일 생성 |

### 0.3 해상도에 따른 패치 수 변경

CheXFound은 512×512 해상도를 사용하므로 Architecture_Final.md의 196개 패치 (224 기준) 대신 1024개 패치가 된다.

| 파라미터 | Architecture_Final.md | CheXFound 기준 |
|---|---|---|
| 이미지 해상도 | 224×224 | 512×512 |
| 패치 수 (global) | 14×14 = 196 | 32×32 = 1024 |
| Phase 2 마스킹 70% | ~137개 마스킹 | ~717개 마스킹 |
| Queue 실효성 | 68,000개 | 여전히 유효 (패치당 많아져 Queue 비율은 낮아짐) |

패치 수가 많아지므로 prototype K=512로 설정한다. Ablation에서 256, 512, 1024을 비교한다.

---

## 1. iBOT Centering 구현

### 1.1 개요

Teacher 출력에 centering을 적용하여 모든 출력이 한 점으로 collapse되는 것을 방지한다. Center vector를 배치 평균의 EMA로 업데이트한다.

### 1.2 구현

```python
class CenterEMA:
    def __init__(self, dim, momentum=0.9):
        self.center = torch.zeros(dim)  # 초기값 0
        self.momentum = momentum
    
    @torch.no_grad()
    def update(self, teacher_output):
        """teacher_output: (B * num_patches, dim)"""
        batch_center = teacher_output.mean(dim=0)
        self.center = self.momentum * self.center + (1 - self.momentum) * batch_center
    
    def apply(self, teacher_output):
        """centering 적용 후 반환"""
        return teacher_output - self.center
```

### 1.3 파라미터

| 파라미터 | 값 | 비고 |
|---|---|---|
| center 차원 | d (= 1024) | z_patch 차원과 동일 |
| EMA momentum | 0.9 | DINOv2 원본과 동일 |
| 초기값 | 0 벡터 | 학습 초기 배치에서 빠르게 수렴 |

### 1.4 적용 위치

Phase 1과 Phase 2 모두에서 iBOT Teacher 출력에 centering을 적용한 후 softmax를 취한다.

```python
# Teacher forward
teacher_patch_tokens = teacher_vit(global_view)  # (B, N, d)
centered = center_ema.apply(teacher_patch_tokens)  # centering
q = softmax(centered / tau_t, dim=-1)  # teacher 타겟

# center 업데이트
center_ema.update(teacher_patch_tokens)
```

---

## 2. Sinkhorn-Knopp 구현

### 2.1 개요

배치 + Queue 전체에 대해 prototype 배정을 균등하게 강제한다. Log-domain에서 연산하여 numerical stability를 확보한다.

### 2.2 구현

```python
@torch.no_grad()
def sinkhorn_knopp(scores, num_iters=3, epsilon=0.05):
    """
    scores: (N_total, K) — 현재 배치 + Queue의 모든 패치와 K개 prototype의 내적 / tau_t
            N_total = batch_patches + queue_size
    num_iters: Sinkhorn 반복 수
    epsilon: softmax smoothing (scores는 이미 tau_t로 나뉜 상태이므로 추가 epsilon 불필요할 수 있음)
    
    Returns: Q (N_total, K) — 균등 배정된 soft assignment
    """
    Q = torch.exp(scores).T  # (K, N_total)
    K_dim, N = Q.shape
    
    # 균등 배정 타겟
    sum_Q = Q.sum()
    Q /= sum_Q
    
    for _ in range(num_iters):
        # Row normalization: 각 prototype이 균등하게 사용되도록
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= K_dim
        
        # Column normalization: 각 패치가 총 확률 1을 갖도록
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= N
    
    Q = Q.T  # (N_total, K)
    Q *= N   # 스케일 복원
    
    return Q
```

### 2.3 Log-domain 안정화 버전

scores 값이 매우 크거나 작을 때 exp overflow/underflow를 방지한다.

```python
@torch.no_grad()
def sinkhorn_knopp_log(scores, num_iters=3):
    """
    scores: (N_total, K) — log-domain에서 연산
    """
    Q = scores.T  # (K, N_total), log-domain
    K_dim, N = Q.shape
    
    for _ in range(num_iters):
        # Row normalization (log-domain)
        Q -= torch.logsumexp(Q, dim=1, keepdim=True)
        Q -= torch.log(torch.tensor(K_dim, dtype=Q.dtype))
        
        # Column normalization (log-domain)
        Q -= torch.logsumexp(Q, dim=0, keepdim=True)
        Q -= torch.log(torch.tensor(N, dtype=Q.dtype))
    
    Q = Q.T  # (N_total, K)
    return torch.exp(Q) * N  # 확률로 변환, 스케일 복원
```

### 2.4 Queue와 함께 사용

```python
# Queue: (queue_size, d) — Teacher z_patch 저장
# current_batch: (B * num_patches, d) — 현재 배치 Teacher z_patch

# 1. 현재 배치 + Queue 합치기
all_patches = torch.cat([current_batch, queue], dim=0)  # (N_total, d)

# 2. Prototype과 내적
scores = torch.mm(all_patches, prototypes.T) / tau_t  # (N_total, K)

# 3. Sinkhorn
Q = sinkhorn_knopp_log(scores, num_iters=3)  # (N_total, K)

# 4. 현재 배치의 타겟만 추출
q_batch = Q[:current_batch.shape[0]]  # (B * num_patches, K)

# 5. Queue 업데이트 (FIFO)
queue = torch.cat([current_batch.detach(), queue], dim=0)[:queue_size]
```

### 2.5 파라미터

| 파라미터 | 값 | 비고 |
|---|---|---|
| num_iters | 3 | SwAV 원본과 동일 |
| Queue 크기 | 65,536 | |
| scores 계산 | z_patch · c_k / τ_t | τ_t = 0.04 |
| 연산 방식 | log-domain 권장 | numerical stability |

---

## 3. Multi-crop Overlap 좌표 역변환

### 3.1 개요

Teacher global view와 Student view (global 또는 local)에서 같은 원본 이미지 위치를 가리키는 패치를 찾아 매칭한다. L_A와 L_R 모두 overlap 패치에서만 계산한다.

### 3.2 좌표 시스템

```
원본 이미지: H_orig × W_orig (예: 512 × 512)
Global crop: (top, left, height, width) → resize to 224 × 224
Local crop:  (top, left, height, width) → resize to 96 × 96
ViT patch:   16 × 16 → global view는 14×14=196 패치, local view는 6×6=36 패치
```

### 3.3 패치 인덱스 → 원본 이미지 좌표 매핑

```python
def patch_to_orig_coords(patch_idx, crop_params, input_size, patch_size=16):
    """
    patch_idx: (row, col) — ViT 패치 그리드에서의 위치
    crop_params: dict with keys 'top', 'left', 'height', 'width'
    input_size: crop을 resize한 후의 크기 (예: 224)
    
    Returns: (orig_top, orig_left, orig_bottom, orig_right) — 원본 이미지에서의 영역
    """
    grid_size = input_size // patch_size  # 예: 224//16 = 14
    
    # 패치가 resize된 crop에서 차지하는 영역
    patch_top_in_crop = patch_idx[0] / grid_size  # 비율 (0~1)
    patch_left_in_crop = patch_idx[1] / grid_size
    patch_bottom_in_crop = (patch_idx[0] + 1) / grid_size
    patch_right_in_crop = (patch_idx[1] + 1) / grid_size
    
    # crop 영역을 원본 이미지 좌표로 변환
    crop_top = crop_params['top']
    crop_left = crop_params['left']
    crop_h = crop_params['height']
    crop_w = crop_params['width']
    
    orig_top = crop_top + patch_top_in_crop * crop_h
    orig_left = crop_left + patch_left_in_crop * crop_w
    orig_bottom = crop_top + patch_bottom_in_crop * crop_h
    orig_right = crop_left + patch_right_in_crop * crop_w
    
    return orig_top, orig_left, orig_bottom, orig_right
```

### 3.4 Overlap 패치 매칭

```python
def find_overlapping_patches(teacher_crop_params, student_crop_params,
                              teacher_input_size=224, student_input_size=224,
                              patch_size=16, iou_threshold=0.5):
    """
    Teacher와 Student의 패치 중 원본 이미지에서 겹치는 쌍을 찾는다.
    
    Returns: list of (teacher_patch_idx, student_patch_idx) 쌍
    """
    teacher_grid = teacher_input_size // patch_size
    student_grid = student_input_size // patch_size
    
    overlaps = []
    
    for t_row in range(teacher_grid):
        for t_col in range(teacher_grid):
            t_coords = patch_to_orig_coords(
                (t_row, t_col), teacher_crop_params, teacher_input_size, patch_size
            )
            
            for s_row in range(student_grid):
                for s_col in range(student_grid):
                    s_coords = patch_to_orig_coords(
                        (s_row, s_col), student_crop_params, student_input_size, patch_size
                    )
                    
                    # IoU 계산
                    inter_top = max(t_coords[0], s_coords[0])
                    inter_left = max(t_coords[1], s_coords[1])
                    inter_bottom = min(t_coords[2], s_coords[2])
                    inter_right = min(t_coords[3], s_coords[3])
                    
                    if inter_top >= inter_bottom or inter_left >= inter_right:
                        continue
                    
                    inter_area = (inter_bottom - inter_top) * (inter_right - inter_left)
                    t_area = (t_coords[2] - t_coords[0]) * (t_coords[3] - t_coords[1])
                    s_area = (s_coords[2] - s_coords[0]) * (s_coords[3] - s_coords[1])
                    union_area = t_area + s_area - inter_area
                    
                    iou = inter_area / (union_area + 1e-6)
                    
                    if iou >= iou_threshold:
                        t_idx = t_row * teacher_grid + t_col
                        s_idx = s_row * student_grid + s_col
                        overlaps.append((t_idx, s_idx))
    
    return overlaps
```

### 3.5 최적화: 사전 계산

위 brute-force 방식은 O(N_t × N_s)로 느리다. 실제 구현에서는 crop 파라미터로부터 overlap 영역을 직접 계산하여 O(1)에 가까운 매칭이 가능하다.

```python
def fast_overlap_indices(teacher_crop, student_crop,
                          teacher_input_size=224, student_input_size=224,
                          patch_size=16):
    """
    Crop 파라미터로부터 overlap 패치 인덱스를 직접 계산한다.
    
    핵심 아이디어: 원본 이미지에서 두 crop의 교집합 영역을 구하고,
    그 영역에 해당하는 패치 인덱스를 각 crop의 그리드에서 역산한다.
    """
    # 두 crop의 원본 이미지 영역
    t_top, t_left = teacher_crop['top'], teacher_crop['left']
    t_bot = t_top + teacher_crop['height']
    t_right = t_left + teacher_crop['width']
    
    s_top, s_left = student_crop['top'], student_crop['left']
    s_bot = s_top + student_crop['height']
    s_right = s_left + student_crop['width']
    
    # 교집합 영역
    inter_top = max(t_top, s_top)
    inter_left = max(t_left, s_left)
    inter_bot = min(t_bot, s_bot)
    inter_right = min(t_right, s_right)
    
    if inter_top >= inter_bot or inter_left >= inter_right:
        return [], []  # 겹치는 영역 없음
    
    # Teacher 그리드에서의 인덱스 범위
    t_grid = teacher_input_size // patch_size
    t_row_start = int((inter_top - t_top) / teacher_crop['height'] * t_grid)
    t_row_end = int((inter_bot - t_top) / teacher_crop['height'] * t_grid)
    t_col_start = int((inter_left - t_left) / teacher_crop['width'] * t_grid)
    t_col_end = int((inter_right - t_left) / teacher_crop['width'] * t_grid)
    
    # Student 그리드에서의 인덱스 범위
    s_grid = student_input_size // patch_size
    s_row_start = int((inter_top - s_top) / student_crop['height'] * s_grid)
    s_row_end = int((inter_bot - s_top) / student_crop['height'] * s_grid)
    s_col_start = int((inter_left - s_left) / student_crop['width'] * s_grid)
    s_col_end = int((inter_right - s_left) / student_crop['width'] * s_grid)
    
    # 인덱스 쌍 생성
    teacher_indices = []
    student_indices = []
    
    t_rows = range(max(0, t_row_start), min(t_grid, t_row_end + 1))
    t_cols = range(max(0, t_col_start), min(t_grid, t_col_end + 1))
    s_rows = range(max(0, s_row_start), min(s_grid, s_row_end + 1))
    s_cols = range(max(0, s_col_start), min(s_grid, s_col_end + 1))
    
    # 대응하는 인덱스 매칭 (비율 기반)
    for t_r in t_rows:
        # Teacher 패치의 원본 이미지 중심점
        t_center_y = t_top + (t_r + 0.5) / t_grid * teacher_crop['height']
        # 이 중심점이 Student 그리드에서 어디에 해당하는지
        s_r = int((t_center_y - s_top) / student_crop['height'] * s_grid)
        if s_r < 0 or s_r >= s_grid:
            continue
            
        for t_c in t_cols:
            t_center_x = t_left + (t_c + 0.5) / t_grid * teacher_crop['width']
            s_c = int((t_center_x - s_left) / student_crop['width'] * s_grid)
            if s_c < 0 or s_c >= s_grid:
                continue
            
            teacher_indices.append(t_r * t_grid + t_c)
            student_indices.append(s_r * s_grid + s_c)
    
    return teacher_indices, student_indices
```

### 3.6 Augmentation 파라미터 저장

DINOv2의 multi-crop transform에서 crop 파라미터를 반환하도록 수정해야 한다.

```python
class MultiCropTransformWithParams:
    """
    DINOv2의 DataAugmentationDINO를 수정하여
    각 crop의 파라미터를 함께 반환한다.
    """
    def __call__(self, image):
        crops = []
        crop_params = []
        
        for scale, size in self.crop_configs:
            # RandomResizedCrop의 파라미터를 직접 생성
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=scale, ratio=(3/4, 4/3)
            )
            
            crop = transforms.functional.resized_crop(
                image, i, j, h, w, (size, size)
            )
            
            # augmentation (color jitter, flip 등) 적용
            crop = self.augment(crop)
            
            crops.append(crop)
            crop_params.append({
                'top': i,
                'left': j,
                'height': h,
                'width': w,
                'output_size': size,
            })
        
        return crops, crop_params
```

### 3.7 주의사항

- Horizontal flip이 적용된 경우 left 좌표를 뒤집어야 한다. flip 여부도 crop_params에 저장할 것.
- Local crop의 input_size가 global crop과 다르면 (예: 96 vs 224) grid_size가 달라지므로 매칭 시 주의.
- 배치 내 이미지마다 crop 파라미터가 다르므로 이미지별로 overlap을 계산해야 한다.

---

## 4. Prototype L2 Normalization (매 step)

```python
class PrototypeDictionary(nn.Module):
    def __init__(self, K, d):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(K, d))
        # k-means 결과로 초기화한 후 normalize
        nn.init.normal_(self.prototypes)
        self.normalize()
    
    @torch.no_grad()
    def normalize(self):
        """매 optimizer step 이후 호출"""
        self.prototypes.data = F.normalize(self.prototypes.data, dim=1)
    
    def forward(self, z_patch):
        """
        z_patch: (B, N, d) — L2 normalized
        Returns: scores (B, N, K)
        """
        z_norm = F.normalize(z_patch, dim=-1)
        return torch.einsum('bnd,kd->bnk', z_norm, self.prototypes)
```

학습 루프에서:

```python
optimizer.step()
prototype_dict.normalize()  # 매 step 후 L2 normalize
```

---

## 5. Residual Projection 구현

```python
def compute_z_A(z_patch, prototype_scores, prototypes):
    """
    z_patch: (B, N, d)
    prototype_scores: (B, N, K) — softmax 이전의 raw scores
    prototypes: (K, d)
    
    Returns: z_A (B, N, d) — prototype soft assignment 가중합
    """
    p = F.softmax(prototype_scores / tau_s, dim=-1)  # (B, N, K)
    z_A = torch.einsum('bnk,kd->bnd', p, prototypes)  # (B, N, d)
    return z_A


def compute_z_R(z_patch, z_A):
    """
    z_patch: (B, N, d)
    z_A: (B, N, d) — stop-gradient 적용 대상
    
    Returns: z_R (B, N, d) — z_A와 직교하는 잔차
    """
    z_A_sg = z_A.detach()  # stop-gradient
    
    # proj_{z_A} z_patch = (z_patch · z_A / ||z_A||^2) * z_A
    dot = (z_patch * z_A_sg).sum(dim=-1, keepdim=True)  # (B, N, 1)
    norm_sq = (z_A_sg * z_A_sg).sum(dim=-1, keepdim=True) + 1e-8  # (B, N, 1)
    proj = (dot / norm_sq) * z_A_sg  # (B, N, d)
    
    z_R = z_patch - proj  # (B, N, d)
    return z_R
```

---

## 6. Dynamic Weighting 구현

```python
def compute_dynamic_weights(ibot_per_patch_loss, tau_w=0.5):
    """
    ibot_per_patch_loss: (M,) — 마스킹된 패치들의 iBOT cross-entropy
    tau_w: weighting temperature
    
    Returns: w (M,) — 정규화된 가중치, 합 = M
    """
    M = ibot_per_patch_loss.shape[0]
    
    # softmax 기반 가중치
    w = torch.exp(ibot_per_patch_loss / tau_w)
    w = w / (w.mean() + 1e-8)  # 평균이 1이 되도록 정규화
    # 합이 M이 되도록
    # (w.mean() = 1이면 w.sum() = M)
    
    return w.detach()  # gradient가 w를 통해 흐르지 않도록
```

**주의:** `w.detach()`로 gradient를 차단한다. w는 L_R의 가중치로만 사용되며 w 자체가 학습되지 않는다.

---

## 7. Phase 2 학습 루프 스케치

```python
# Phase 2 학습 루프 (pseudocode)
for epoch in range(phase2_epochs):
    for batch in dataloader:
        # 1. Multi-crop 생성
        (teacher_global_views, teacher_crop_params,
         student_views, student_crop_params) = multi_crop_transform(batch)
        
        # 2. Teacher forward (no grad)
        with torch.no_grad():
            teacher_patches = teacher_vit(teacher_global_views)  # (B, N, d)
        
        # 3. Student forward
        student_patches = student_vit(student_views)  # (B, N', d), 마스킹 적용됨
        
        # 4. iBOT loss (z_patch 수준, 마스킹된 패치)
        ibot_loss, per_patch_ibot_loss = compute_ibot_loss(
            student_patches, teacher_patches, masked_indices
        )
        
        # 5. Prototype assignment (L_A용)
        scores_teacher = prototype_dict(teacher_patches)  # (B, N, K)
        scores_student = prototype_dict(student_patches)  # (B, N', K)
        
        # Sinkhorn on teacher (with queue)
        q_teacher = sinkhorn_with_queue(scores_teacher, queue)
        p_student = F.softmax(scores_student / tau_s, dim=-1)
        
        # L_A (overlap 패치만)
        overlap_t, overlap_s = fast_overlap_indices(teacher_crop_params, student_crop_params)
        la_loss = cross_entropy(q_teacher[overlap_t], p_student[overlap_s])
        
        # 6. z_A, z_R 계산
        z_A_teacher = compute_z_A(teacher_patches, scores_teacher, prototype_dict.prototypes)
        z_A_student = compute_z_A(student_patches, scores_student, prototype_dict.prototypes)
        
        z_R_teacher = compute_z_R(teacher_patches, z_A_teacher)
        z_R_student = compute_z_R(student_patches, z_A_student)
        
        # 7. L_R (마스킹된 패치만)
        lr_per_patch = 1 - F.cosine_similarity(
            z_R_student[masked_indices],
            z_R_teacher[masked_indices].detach(),
            dim=-1
        )
        
        # 8. Dynamic weighting
        if epoch < warmup_epochs:
            w = torch.ones_like(lr_per_patch)
        else:
            w = compute_dynamic_weights(per_patch_ibot_loss[masked_indices])
        
        lr_loss = (w * lr_per_patch).mean()
        
        # 9. Total loss
        loss = ibot_loss + lambda_R * lr_loss + lambda_A_P2 * la_loss
        
        # 10. Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prototype_dict.normalize()
        
        # 11. Teacher EMA update
        ema_update(teacher_vit, student_vit, momentum=0.9995)
        
        # 12. Queue update
        queue_update(teacher_patches.detach())
```

---

## 8. 주의사항 요약

1. **z_patch L2 normalize:** prototype과 내적 전에 z_patch를 L2 normalize해야 cosine similarity 기반 비교가 된다.
2. **Prototype L2 normalize:** 매 optimizer step 후 prototype을 L2 normalize한다.
3. **stop-gradient 위치:** z_A 계산 시 `.detach()`, Teacher z_R 계산 시 `.detach()`, dynamic weight w 계산 시 `.detach()`.
4. **Queue는 Teacher z_patch만 저장:** Student 것은 매 iteration 급격히 변해서 queue에 부적합.
5. **Phase 2에서 Prototype C는 frozen:** `prototype_dict.prototypes.requires_grad = False` 또는 optimizer에서 제외.
6. **Phase 2에서 L_A gradient 경로:** C가 frozen이므로 gradient는 scores → z_patch → ViT로만 흐른다.
7. **Overlap 매칭 시 flip 고려:** horizontal flip이 적용된 경우 좌표를 뒤집어야 한다.
8. **Masking ratio:** Phase 1은 (0.1, 0.5) 랜덤, Phase 2는 0.7 고정.
