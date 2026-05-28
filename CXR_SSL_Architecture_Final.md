# DINOv2 기반 CXR 병변 Segmentation을 위한 Self-Supervised Learning 아키텍처

## 1. 문제 정의

DINOv2는 CXR에서 anatomy 수준의 분리(폐, 갈비뼈, 횡격막 등)는 잘 수행하지만, 동일 해부학 영역 내에서 정상 조직과 병변을 구분하지 못한다. PCA 시각화에서 폐 영역 내 모든 패치가 하나의 cluster로 뭉개지는 현상이 관찰된다.

이는 CLS self-distillation의 1 image 1 class 가정에 기인한다. 같은 이미지의 서로 다른 위치 crop CLS를 일치시키는 압력이 병변 패치가 자기만의 cluster를 형성하는 것을 방해한다.

본 연구는 DINOv2의 Teacher-Student EMA 구조와 iBOT masked patch prediction을 보존하면서, CLS self-distillation을 patch-level prototype learning으로 대체하여 병변 패치가 정상 패치와 분리되는 표현 공간을 학습한다.

### 핵심 가설

1. **Phase 1 가설:** Sinkhorn 균등 배정으로 학습된 prototype이 데이터셋의 공통 해부학 패턴을 흡수한다. 병변은 소수이므로 prototype을 독점하지 못한다.
2. **Phase 2 가설:** Prototype으로 설명되지 않는 직교 성분(z_R)이 병변 신호를 담는다. Reconstruction error가 큰 패치가 병변 후보이다.

두 가설 모두 귀납적이며, 이를 실험적으로 검증하는 것이 본 연구의 핵심 contribution이다.

---

## 2. 전체 아키텍처 개요

### 2.1 기본 골격

| 구성 요소 | 사양 |
|---|---|
| Backbone | ViT-L/14 (DINOv2 pretrained weight으로 초기화) |
| Patch size | 14×14 |
| Embedding dim (d) | 1024 |
| Teacher 업데이트 | EMA, momentum m = 0.9995 |
| Global crop 수 | 2 (Teacher, Student 각각) |
| Local crop 수 | Phase 1: 4 / Phase 2: 8 (Student만, random masking 적용) |
| Global crop scale | (0.4, 1.0) |
| Local crop scale | Phase 1: (0.20, 0.5) / Phase 2: (0.05, 0.4) |
| Masking ratio | Phase 1: (0.1, 0.5) / Phase 2: 0.7 |

### 2.2 입력 구성

DINOv2의 multi-crop 전략을 그대로 유지한다.

- **Teacher:** Global crop 2장을 받아 마스킹 없이 forward pass.
- **Student:** Global crop 2장 + Local crop (Phase 1: 4장, Phase 2: 8장)을 받아 random masking 적용 후 forward pass.
- Teacher는 Student의 EMA로 업데이트된다.

### 2.3 학습 단계

학습은 두 단계로 나뉜다.

- **Phase 1:** Prototype을 수렴시켜 데이터셋의 공통 해부학 패턴을 정의한다. z_R은 비활성 상태이다.
- **Phase 2:** Prototype이 안정화된 후, z_patch에서 prototype 방향 성분을 제거한 직교 성분 z_R을 학습한다.

Phase 1 검증을 통과해야만 Phase 2로 진입한다.

---

## 3. Phase 1 — Prototype 수렴

### 3.1 목적

ViT encoder와 prototype dictionary를 수렴시켜 데이터셋의 공통 해부학 패턴을 정의한다.

### 3.2 Prototype Dictionary

| 파라미터 | 값 |
|---|---|
| Prototype 수 (K) | 512 |
| Prototype 차원 | d (= 1024), z_patch와 동일 |
| 초기화 | PCA 초기화 (Teacher 첫 배치 z_patch의 상위 K 주성분) |
| 배정 방식 | Sinkhorn-Knopp soft assignment |

**초기화 방법:**

학습 시작 시 Teacher backbone의 첫 배치 z_patch로 SVD를 수행하여 상위 K개 주성분 방향을 prototype 초기값으로 사용한다.

1. Teacher backbone으로 첫 배치의 global crop을 forward한다 (stop-gradient).
2. 추출된 z_patch를 L2 normalize한 뒤 최대 K×10 = 5,120개로 서브샘플한다.
3. 평균 중심화 후 SVD를 수행하여 상위 K개 오른쪽 특이벡터(Vᴴ[:K])를 추출한다.
4. L2 normalize 후 prototype dictionary C의 초기값으로 설정한다.
5. rank 0에서 계산한 결과를 모든 GPU로 broadcast하여 동기화한다.

> **설계 변경 이유 (k-means → PCA):**
> DINOv2 pretrained encoder의 z_patch로 k-means (K=512)를 수행한 결과, 모든 prototype vector 간 pairwise cosine similarity가 0.98 수준으로 나타났다. CXR patch 임베딩이 고차원 공간의 좁은 영역에 밀집되어 있어 k-means center가 사실상 같은 방향을 가리키는 현상이다.
>
> PCA 초기화는 두 가지 장점을 제공한다. 첫째, 주성분 방향은 구조적으로 직교하므로 초기 pairwise similarity ≈ 0이 보장된다. 둘째, 실제 patch 분산의 주요 축을 포착하므로 랜덤 초기화보다 의미 있는 방향에서 학습이 시작된다. Sinkhorn의 균등 배정 압력이 서로 다른 방향의 prototype을 각자의 해부학 패턴으로 수렴시키는 데 훨씬 유리하다.

**업데이트 방식:**

C는 learnable parameter (nn.Parameter)로 선언하고 optimizer에 포함시켜 L_A의 backprop으로 업데이트한다. 매 step 이후 각 prototype vector를 L2 normalize하여 단위구 위에서만 방향이 학습되도록 한다. Sinkhorn 균등 배정이 모든 prototype이 비슷한 빈도로 사용되도록 강제하므로 데이터 분포의 대표 방향으로 수렴한다.

Prototype과 z_patch가 같은 1024차원에서 직접 비교된다. 별도의 projection head 없이 내적으로 유사도를 계산한다.

### 3.3 Prototype Assignment

**Teacher (타겟 생성):**

1. Teacher의 z_patch를 L2 normalize한다.
2. z_patch와 prototype dictionary C의 내적을 계산한다.

$$s_k = z_{\text{patch}}^T \cdot c_k / \tau_t$$

3. Sinkhorn-Knopp 알고리즘 (3회 반복)으로 **현재 배치 + Queue** 전체에 대해 균등 배정을 강제한 타겟 분포 q를 구한다.

**Student (예측):**

1. Student의 z_patch를 L2 normalize한다.
2. softmax로 예측 분포 p를 구한다.

$$p_k = \frac{\exp(z_{\text{patch}}^S \cdot c_k / \tau_s)}{\sum_j \exp(z_{\text{patch}}^S \cdot c_j / \tau_s)}$$

**Temperature 설정:**

| 파라미터 | 값 | 비고 |
|---|---|---|
| τ_t (Teacher) | 0.04 | Teacher 분포를 sharp하게 유지 |
| τ_s (Student) | 0.1 | Student 분포는 상대적으로 soft |

### 3.4 Queue

| 파라미터 | 값 |
|---|---|
| Queue 크기 | 131,072 patch token |
| 저장 대상 | Teacher의 z_patch (L2-normalized, stop-gradient, fp16 저장) |
| 업데이트 방식 | FIFO — 매 배치마다 Teacher z_patch 전체를 enqueue, 오래된 것 dequeue |

Sinkhorn 균등 배정은 배치 단위로 작동하므로 배치 구성에 따라 prototype 배정이 흔들릴 수 있다. Queue를 도입하면 현재 배치 patch token (~49,000개, GPU당 2B×N_g) + Queue(131,072개)를 합쳐 총 ~180,000개 patch token에 대해 Sinkhorn을 수행하여 더 안정적인 배정이 가능하다. Queue에는 Teacher의 z_patch token을 그대로 저장하므로 현재 배치와 동일한 분포를 유지한다 (image mean 같은 이종 벡터를 섞지 않는다).

### 3.5 Overlap 처리

Teacher global view와 Student view가 겹치는 위치의 패치만 L_A에 참여한다. Augmentation 파라미터(crop 좌표, 리사이즈 비율)를 저장해두고 패치 인덱스를 원본 이미지 좌표로 역변환하여 overlap 영역을 계산한다.

### 3.6 Phase 1 Loss

$$L_{\text{Phase1}} = L_{\text{iBOT}} + \lambda_A \cdot L_A$$

**L_iBOT (Masked Patch Prediction):**

DINOv2 원본과 동일하다. z_patch 수준에서 작동한다. Student의 마스킹된 패치 위치에서 Teacher의 해당 위치 z_patch 출력을 타겟으로 cross-entropy loss를 계산한다. Phase 1과 Phase 2 모두 동일하게 z_patch 기반으로 수행한다 (z_R 기반이 아님).

$$L_{\text{iBOT}} = -\sum_{i \in \text{masked}} \sum_k q_k^{(i)} \log p_k^{(i)}$$

여기서 q는 Teacher z_patch 출력의 softmax (centering 적용), p는 Student z_patch 출력의 softmax이다.

**L_A (Prototype Assignment Loss):**

Teacher의 Sinkhorn 타겟 q와 Student의 softmax 예측 p 사이의 cross-entropy이다.

$$L_A = -\frac{1}{|P|} \sum_{i \in P} \sum_{k=1}^{K} q_k^{(i)} \log p_k^{(i)}$$

여기서 P는 overlap 패치 인덱스 집합이다.

**Gradient 차단 설계:**

L_A 계산 시 Student의 z_patch에 `stop-gradient`를 적용한다. 즉, L_A의 gradient는 `prototype_dict.prototypes`만 업데이트하고 ViT backbone으로 역전파되지 않는다.

```python
# Student z_patch는 detach — L_A는 prototype만 학습
sg_norm = F.normalize(student_z_patch_global.detach().float(), dim=-1)
sl_norm = F.normalize(student_z_patch_local.detach().float(), dim=-1)
```

이 설계 없이는 L_A의 gradient가 iBOT gradient와 충돌하여 la_loss가 log(K) = 6.238에서 plateau되고 prototype이 전혀 수렴하지 않는다 (§10 실험 이력 참조). detach 이후 L_A gradient 목적지는 prototype만으로 한정되므로 lambda_A 크기에 무관하게 iBOT 학습이 간섭받지 않는다.

**가중치:**

| 가중치 | 값 | 비고 |
|---|---|---|
| L_iBOT | 1.0 (기준) | |
| λ_A | 0.1 (warmup: 0 → 0.1, 50 epochs) | Student detach 이후 iBOT 간섭 없음 |

### 3.7 Phase 1 학습 설정

| 파라미터 | 값 |
|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.999) |
| Learning rate | 2e-4 (sqrt scaling, cosine schedule, linear warmup 10 epochs) |
| Weight decay | 0.04 고정 |
| Batch size | 14 per GPU |
| Epoch 수 | 100 |
| 수렴 기준 | 배치 간 prototype 배정 분포 변화 < 0.01 (10 epoch 이동 평균) |

### 3.8 Phase 1 검증 (Phase 2 진입 조건)

Phase 1 완료 후 다음 실험을 수행한다. 기대 결과에 미치지 못하면 Phase 2로 진행하지 않고 설계를 재검토한다.

| 실험 | 방법 | 기대 결과 |
|---|---|---|
| Prototype t-SNE/PCA | Prototype vector C를 시각화 | 해부학 구조별 cluster 형성 |
| Patch assignment 시각화 | CXR 이미지 위에 각 패치의 dominant prototype을 색으로 표시 | 해부학 영역별 색 분리 |
| Patch token 품질 | DINOv2 원본 vs Phase 1 모델의 patch token linear probe 비교 | 성능 유지 또는 개선 |

---

## 4. Phase 2 — Residual 학습

### 4.1 목적

Prototype으로 설명되는 공통 성분(z_A)을 z_patch에서 제거하고, 남은 직교 성분(z_R)이 병변 신호를 담도록 학습한다.

### 4.2 Phase 전환 설정

Phase 1 완료 후 Prototype C를 freeze한다. ViT encoder는 freeze하지 않는다. Queue는 Phase 2에서도 유지하여 L_A의 Sinkhorn 배정 안정성을 보장한다.

| 구성 요소 | Phase 2 상태 |
|---|---|
| ViT encoder (Student) | 업데이트 |
| ViT encoder (Teacher) | EMA 업데이트 |
| Prototype C | Freeze |
| Queue | 유지 (L_A Sinkhorn용) |

### 4.3 Phase 2 마스킹

Phase 2에서는 마스킹 비율을 Phase 1보다 높게 설정한다.

| 파라미터 | Phase 1 | Phase 2 |
|---|---|---|
| Masking ratio | (0.1, 0.5) | 0.7 (고정) |

Phase 1에서 이미 충분한 복원 능력을 갖추었으므로 더 높은 마스킹에서도 복원이 가능하다. 마스킹 비율을 높이는 이유는 두 가지이다.

첫째, 더 많은 패치에서 e_i가 계산되어 병변 패치가 마스킹될 확률이 올라간다. 196개 패치 중 약 137개가 마스킹되어 z_R 학습 후보군이 넓어진다.

둘째, 적은 맥락으로도 복원해야 하므로 더 robust한 표현이 학습된다.

### 4.4 z_A 정의

한 패치의 z_A는 해당 패치의 prototype soft assignment 가중합으로 정의한다.

$$z_A = \sum_{k=1}^{K} p_k \cdot c_k$$

여기서 $p_k$는 softmax assignment 확률, $c_k$는 prototype vector이다.

한 패치 안에 여러 anatomy가 겹쳐있으면 (예: 갈비뼈 + 폐혈관) z_A가 해당 prototype들의 가중합이 된다. 복합 anatomy도 prototype 선형 결합으로 흡수되어 z_R에 남지 않는다.

### 4.5 Residual Projection

$$z_R = z_{\text{patch}} - \text{sg}\left(\text{proj}_{z_A} z_{\text{patch}}\right)$$

풀어 쓰면:

$$\text{proj}_{z_A} z_{\text{patch}} = \frac{z_{\text{patch}} \cdot z_A}{\|z_A\|^2} \cdot z_A$$

$$z_R = z_{\text{patch}} - \frac{z_{\text{patch}} \cdot z_A}{\|z_A\|^2} \cdot z_A$$

**설계 원칙:**

- **sg (stop-gradient):** z_R 계산 시 z_A 방향으로 gradient가 역류하지 않도록 차단한다.
- **MLP 없음:** Projection 결과를 그대로 z_R로 사용한다. 비선형 변환을 거치면 z_A와의 선형 독립성이 깨질 수 있다.
- **선형 독립성 구조적 보장:** z_R은 수학적으로 z_A와 완벽하게 직교한다. Forward pass 구조의 hard constraint이다.
- **1024차원 유지:** z_patch와 동일 차원에서 residual을 구하므로 정보 손실이 없다.

### 4.6 Teacher z_R

Teacher도 동일한 residual projection으로 z_R을 생성한다. Prototype C가 공유되고 frozen이므로 Teacher와 Student가 같은 기준으로 z_A를 구하고 같은 방식으로 z_R을 구한다.

Teacher ViT는 Phase 2에서도 Student의 EMA (m=0.9995)로 계속 업데이트된다. Teacher z_R은 EMA 특성상 천천히 변하므로 안정적인 타겟 역할을 한다.

### 4.7 z_A 안정성

Phase 2에서 ViT encoder가 업데이트되면 z_patch가 변하고 z_A도 따라 변한다. 이를 두 가지 장치로 완화한다.

**EMA Teacher:** Teacher ViT가 Student의 EMA로 천천히 변하므로 Teacher의 z_A 변화가 극도로 느리다. 타겟 z_R이 안정적으로 유지된다.

**L_A 유지 (낮은 weight):** Phase 2에서 L_A를 λ_A^P2 = 0.1로 유지하여 prototype 배정 구조가 크게 깨지지 않도록 부드러운 제약을 건다. L_R이 dominant하게 작용하고 L_A는 regularizer 수준에 머문다.

### 4.8 Global View → Local View 구조

Phase 1과 동일한 Teacher-Student multi-crop 구조를 z_R 학습에도 적용한다.

- Teacher가 global view에서 z_R을 생성한다 (넓은 맥락에서의 병변 신호).
- Student가 local view의 overlap 패치에서 동일한 z_R을 예측한다 (좁은 맥락에서 병변 신호를 잡아내도록 강제).
- Overlap 처리는 Phase 1과 동일하다.

### 4.9 L_R (Residual Loss)

L_R은 **마스킹된 패치에서만** 계산한다. Student의 z_R이 Teacher의 z_R을 예측하도록 cosine similarity loss를 사용한다.

$$L_R^{(i)} = 1 - \cos\left(z_R^{\text{Student}(i)},\ \text{sg}(z_R^{\text{Teacher}(i)})\right), \quad i \in \text{masked}$$

마스킹된 패치에서만 L_R을 계산하는 이유는 두 가지이다. 첫째, 마스킹되지 않은 패치에서 L_R을 계산하면 Student가 Teacher를 단순 복사하는 trivial solution에 빠질 수 있다. 둘째, iBOT의 reconstruction error e_i가 마스킹된 패치에서만 정의되므로 L_R과 dynamic weighting의 적용 범위가 일치한다.

매 iteration마다 마스킹 위치가 랜덤하게 바뀌므로 epoch 전체로 보면 모든 패치가 골고루 L_R 학습에 참여한다.

### 4.10 Dynamic Weighting

**Warmup 기간 (Phase 2 시작 후 20 epoch):**

$$w_i = 1 \quad (\text{모든 마스킹된 패치 균등})$$

z_R 공간 전체가 최소한의 표현력을 갖도록 한다.

**Warmup 이후:**

iBOT의 z_patch 복원 난이도를 기반으로 가중치를 동적으로 계산한다. e_i는 iBOT의 마스킹된 패치별 cross-entropy이다. z_patch의 복원 난이도가 "이 패치가 주변 맥락으로 예측하기 어려운가"를 직접 반영한다.

$$e_i = L_{\text{iBOT}}^{(i)}, \quad i \in \text{masked}$$

$$w_i = \frac{\exp(e_i / \tau_w)}{\frac{1}{M}\sum_{j \in \text{masked}} \exp(e_j / \tau_w)}$$

- $e_i$: 마스킹된 패치 i의 iBOT cross-entropy
- $\tau_w$: weighting temperature (기본값 0.5)
- M: 배치 내 마스킹된 패치 수
- 정규화로 가중치 합이 항상 M으로 유지

복원 쉬운 패치(정상)는 iBOT cross-entropy가 작아 $w_i$가 작고 L_R 영향이 미미하다. z_R은 노이즈 수준으로 남는다. 복원 어려운 패치(병변 후보)는 iBOT cross-entropy가 커서 $w_i$가 크고 L_R이 강하게 작용한다. z_R이 병변 신호를 학습한다.

### 4.11 Phase 2 Loss

$$L_{\text{Phase2}} = L_{\text{iBOT}} + \lambda_R \cdot \frac{1}{M}\sum_{i \in \text{masked}} w_i \cdot L_R^{(i)} + \lambda_A^{P2} \cdot L_A$$

| 가중치 | 값 | 비고 |
|---|---|---|
| L_iBOT | 1.0 | Phase 1과 동일 |
| λ_R | 1.0 | cosine similarity 스케일이 iBOT과 유사 |
| λ_A^P2 | 0.1 | z_A 구조 유지를 위한 부드러운 제약 |
| τ_w | 0.5 | dynamic weighting temperature |

iBOT은 z_patch 수준에서 마스킹된 패치를 복원하는 역할과 함께 reconstruction error e_i를 dynamic weighting에 제공한다. L_A는 Prototype C가 frozen이므로 gradient가 z_patch → ViT 방향으로만 흐르며, anatomy 배정 구조를 유지하는 약한 regularizer로 작동한다.

### 4.12 Phase 2 학습 설정

| 파라미터 | 값 |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-5 (cosine schedule) |
| Weight decay | 0.04 → 0.4 (cosine schedule) |
| Batch size | Phase 1과 동일 |
| Epoch 수 | 50 |
| Warmup (dynamic weighting) | 20 epochs |
| Masking ratio | 0.7 |

---

## 5. Gradient 흐름

Phase 2에서 ViT encoder로 전달되는 gradient:

```
L_iBOT  → ViT 업데이트 (맥락 복원 압력)
L_R     → ∂L_R/∂z_R → ∂z_R/∂z_patch → ViT 업데이트 (잔차 학습 압력)
L_A     → prototype_dict 업데이트만 (Student z_patch에 stop-gradient)
         Phase 1과 동일하게 ViT backbone으로 역전파 없음
         (C는 frozen이므로 gradient가 C를 통해 흐르지 않음)
```

z_R = z_patch - sg(proj_{z_A} z_patch) 에서 sg로 인해 projection 항은 상수 취급된다.

$$\frac{\partial z_R}{\partial z_{\text{patch}}} = I$$

z_R의 gradient가 z_patch에 그대로 전달된다.

L_iBOT과 L_R은 상호보완적이다. iBOT이 잘 복원하는 부분은 z_A가 흡수하고, 못 복원하는 부분이 z_R로 흘러간다. L_A는 낮은 weight (0.1)로 L_R과의 충돌을 최소화하면서 z_A 구조를 유지하는 regularizer 역할을 한다.

---

## 6. Downstream 활용

본 방법론의 출력은 각 패치에 대한 두 종류의 임베딩이다.

- **z_A** (1024차원): 해당 패치의 해부학적 맥락. prototype soft assignment 가중합.
- **z_R** (1024차원): 해당 패치에서 anatomy로 설명되지 않는 잔차 신호.

이 두 임베딩은 downstream task에 따라 다양한 방식으로 조합할 수 있다. 아래는 대표적 활용 예시이다.

### 6.1 Segmentation Fine-tuning

#### 6.1.1 전체 파이프라인

```
입력 이미지
  → ViT encoder (freeze) → z_patch
  → Prototype assignment → z_A (anatomy 맥락)
  → Residual projection  → z_R (병변 신호)
  → Fusion module → z_fused
  → Segmentation decoder → mask
```

ViT encoder와 prototype C는 freeze하고, fusion module과 decoder만 학습한다. Residual projection은 연산만 수행하며 학습 파라미터가 없다.

#### 6.1.2 z_A와 z_R의 Fusion

z_R을 primary feature로, z_A를 맥락 정보로 사용한다. z_R이 "이 패치에 이상한 게 있다"는 신호를 담고 z_A가 "이 패치가 해부학적으로 어디인지"를 알려준다.

Fusion 방식은 downstream decoder 구조에 따라 선택한다. 대표적 방식은 cross-attention이다. z_R을 query, z_A를 key/value로 사용하면 z_R이 z_A의 해부학 맥락을 참조하면서 병변 경계를 더 정확하게 추론할 수 있다.

$$Q = W_Q \cdot z_R, \quad K = W_K \cdot z_A, \quad V = W_V \cdot z_A$$

$$z_{\text{fused}} = z_R + \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

Residual connection으로 z_R의 원래 정보를 보존하면서 z_A 맥락을 추가한다.

단순 concat ([z_A ; z_R] → 2048차원)이나 element-wise addition도 가능하며, task 특성에 따라 최적 방식이 달라질 수 있다.

#### 6.1.3 학습 설정

| 구성 요소 | 상태 |
|---|---|
| ViT encoder | Freeze |
| Prototype C | Freeze |
| Residual projection | 연산만, 학습 파라미터 없음 |
| Fusion module | 학습 |
| Segmentation decoder | 학습 |

| 파라미터 | 값 |
|---|---|
| Loss | Dice + Cross-entropy |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Epoch | 데이터셋에 따라 조정 |

### 6.2 Zero-shot Anomaly Detection

학습 없이 z_R의 노름을 anomaly score로 직접 사용한다.

$$\text{anomaly\_score}(i) = \|z_R^{(i)}\|_2$$

각 패치 위치의 anomaly score를 이미지 크기로 업샘플링하여 heatmap을 생성한다.

---

## 7. 구조적 가정과 완화 방안

### 7.1 가정

1. **Prototype → anatomy:** 데이터셋 다수 패턴이 정상 해부학이라는 귀납적 가정. 특정 병변이 데이터셋에 다수 존재하면 prototype이 해당 병변을 흡수할 수 있다.
2. **Reconstruction error → 병변:** 복원 어려운 패치가 병변이라는 가정. 특이한 촬영 각도, 이식 기기, 노이즈도 복원이 어려울 수 있다.

### 7.2 완화 방안

- DINOv2 pretrain init + k-means 초기화로 prototype이 anatomy 방향에서 시작
- Sinkhorn + Queue (65,536)로 안정적 균등 배정
- Phase 구조로 prototype 안정화 후 z_R 학습
- EMA + L_A (낮은 weight)로 Phase 2에서 z_A 안정성 유지
- Warmup으로 z_R 공간의 최소 표현력 확보 후 dynamic weighting 전환

---

## 8. 실험 계획

### 8.1 Phase 1 검증 (Phase 2 진입 조건)

| 실험 | 방법 | 기대 결과 |
|---|---|---|
| Prototype t-SNE/PCA | Prototype vector C 시각화 | 해부학 구조별 cluster 형성 |
| Patch assignment 시각화 | CXR 위에 dominant prototype 색 표시 | 해부학 영역별 색 분리 |
| Patch token 품질 | DINOv2 원본 vs Phase 1 모델 linear probe | 성능 유지 또는 개선 |

### 8.2 Phase 2 검증 및 Ablation

| 우선순위 | 실험 | 목적 |
|---|---|---|
| 1 | z_R heatmap vs GT segmentation | z_R 노름이 병변 위치와 align되는지 |
| 2 | DINOv2 원본 vs 본 방법론 segmentation | 전체 방법론 효과 |
| 3 | Residual projection vs 직교성 제약 | 구조적 분리의 효과 |
| 4 | Dynamic weighting 유무 | 가중치의 효과 |
| 5 | K값 sensitivity (256, 512, 1024) | Prototype 수의 영향 |
| 6 | Phase 구조 유무 | Phase 없이 한번에 학습 시 비교 |
| 7 | λ_A^P2 sensitivity (0.05, 0.1, 0.2) | Phase 2 anatomy 제약 강도 |
| 8 | Phase 2 masking ratio sensitivity (0.5, 0.6, 0.7, 0.8) | Phase 2 마스킹 비율의 영향 |

---

## 9. 하이퍼파라미터 요약

| 파라미터 | 값 | 카테고리 | 비고 |
|---|---|---|---|
| ViT backbone | ViT-L/14 | 아키텍처 | |
| Embedding dim (d) | 1024 | 아키텍처 | |
| Prototype 수 (K) | 512 | 아키텍처 | |
| Prototype 초기화 | PCA (첫 배치 z_patch 상위 K 주성분) | 아키텍처 | k-means 폐기 (§3.2 참조) |
| EMA momentum (m) | 0.9995 | 학습 | |
| Teacher temperature (τ_t) | 0.04 | 학습 | |
| Student temperature (τ_s) | 0.1 | 학습 | |
| Sinkhorn 반복 수 | 3 | 학습 | |
| Queue 크기 | 131,072 | 학습 | patch token 저장 기준 (≈2.7 step 분량) |
| Global crop scale | (0.4, 1.0) | 데이터 | |
| Local crop scale (Phase 1) | (0.20, 0.5) | 데이터 | overlap 안정성 위해 조정 |
| Local crop scale (Phase 2) | (0.05, 0.4) | 데이터 | |
| Global crop 수 | 2 | 데이터 | |
| Local crop 수 (Phase 1) | 4 | 데이터 | 4×196 ≈ 8×100, 효율 유지 |
| Local crop 수 (Phase 2) | 8 | 데이터 | |
| Phase 1 masking ratio | (0.1, 0.5) | 데이터 | |
| Phase 2 masking ratio | 0.7 | 데이터 | |
| Phase 1 epochs | 100 | 학습 | |
| Phase 1 LR | 2e-4 (sqrt scaling) | 학습 | |
| Phase 1 LR warmup | 10 epochs | 학습 | |
| Phase 1 λ_A | 0.1 (warmup 50 epochs) | Loss | 원안 1.0 → iBOT 방해 임계 0.15 기준 조정 |
| Phase 1 weight decay | 0.04 (고정) | 학습 | |
| Phase 2 epochs | 50 | 학습 | |
| Phase 2 LR | 5e-5 | 학습 | |
| Phase 2 LR warmup | 5 epochs | 학습 | |
| Phase 2 dynamic weighting warmup | 20 epochs | 학습 | |
| Phase 2 λ_R | 1.0 | Loss | |
| Phase 2 λ_A^P2 | 0.1 | Loss | |
| Phase 2 weight decay | 0.04 → 0.4 (cosine) | 학습 | |
| τ_w (dynamic weighting) | 0.5 | Loss | |
| Optimizer | AdamW (β1=0.9, β2=0.999) | 학습 | |
| Batch size | 14 per GPU | 학습 | |

---

## 10. 실험 이력

### phase1_prototype (v1) — 중단

- **문제:** weight decay schedule 버그로 학습이 정상적으로 진행되지 않아 640 iter에서 중단.
- **원인:** cosine weight decay가 실제로 작동하지 않아 optimizer 상태가 비정상. 이 상태에서는 다른 하이퍼파라미터 효과를 판단할 수 없어 중단.

### phase1_prototype_v2 — 중단

- **수정 (v1 → v2):** weight decay schedule 버그 수정 (cosine decay 정상 작동). lambda_A 최대값 0.1로 보수적 설정, lambda_A_warmup_epochs=50 도입.
- **결과:** iter 34,020까지 학습. ibot_loss가 최저 3.77 (iter ~22,000) 이후 4.09까지 다시 상승. la_loss는 6.238~6.308 범위에서 plateau → prototype 학습 실패.
- **원인 분석:** L_A에서 Student z_patch로의 gradient 역전파가 존재 → lambda_A × lr이 커지는 후반부에 iBOT과 gradient conflict 발생. la_loss 하한이 log(K) = 6.238이어서 prototype이 전혀 수렴하지 않음.

### phase1_prototype_v3 — 진행 중

- **수정 (v2 → v3):** `PrototypeAssignmentLoss.forward()`에서 Student z_patch에 `.detach()` 추가. L_A gradient가 prototype_dict만 업데이트하고 backbone으로 역전파되지 않도록 차단.
- **기대 결과:** la_loss가 log(512) = 6.238 아래로 실제 감소하면서 prototype 수렴 확인. ibot_loss 상승 없이 안정적 학습 진행.
- **판단 기준:** 학습 초반 100 iter 내에 la_loss가 6.23 아래로 내려가기 시작하면 정상 작동.
