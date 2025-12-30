# 🤖 SSbry - AI 학습 모델

**YOLOv8 기반 스마트 분리배출 AI 모델**

실시간 쓰레기 탐지 및 6종 분류를 위한 커스텀 객체 인식 모델

---

## 📖 모델 개요

### YOLOv8을 선택한 이유

YOLOv8n (Nano) 모델을 기반으로 쓰레기 분류에 최적화된 커스텀 모델을 개발했습니다.

- **실시간 탐지**: 모바일 환경에서도 빠른 추론 속도
- **높은 정확도**: 작은 모델 크기 대비 우수한 분류 성능
- **다중 객체 탐지**: 한 이미지에서 여러 쓰레기 동시 분류 가능
- **쉬운 배포**: ONNX 변환을 통한 Flutter 앱 원활한 연동
- **경량화**: YOLOv8n으로 모바일 최적화

### 시스템 흐름도

```
[원본 이미지]
    ↓
[OpenCV 전처리]
    ├─→ 객체 영역 자동 추출 (Contour Detection)
    ├─→ 배경 노이즈 제거 (Morphology)
    └─→ 정규화 및 리사이징
        ↓
[데이터 증강]
    ├─→ 밝기 변화 (5단계)
    ├─→ 가우시안 노이즈
    ├─→ 회전 및 반전
    └─→ 스케일 변화
        ↓
[YOLO 라벨링]
    └─→ (center_x, center_y, width, height)
        ↓
[YOLOv8n 학습]
    ├─→ 30 Epochs
    ├─→ Batch Size: 4
    └─→ Image Size: 416x416
        ↓
[best_waste_model.pt]
    ↓
[ONNX 변환]
    └─→ Opset 17
        ↓
[모바일 배포]
```

---

## 🗂️ 데이터셋 구성

### 분류 카테고리 (6종)

| 카테고리 | 영문명  |    설명    |         예시          |
| :------: | :-----: | :--------: | :-------------------: |
|    🥫    |   can   |    캔류    |  알루미늄 캔, 철 캔   |
|    🍾    |  glass  |    유리    |   유리병, 유리 용기   |
|    📄    |  paper  |    종이    |   신문지, 박스, 책    |
|    ♻️    | plastic |  플라스틱  | 페트병, 플라스틱 용기 |
|    📦    |  vinyl  |    비닐    |   비닐봉지, 포장재    |
|    🗑️    |  trash  | 일반쓰레기 |  재활용 불가 폐기물   |

### 데이터 출처

- **TrashNet Dataset** (Kaggle)
- 클래스당 평균 200 + 이미지
- 총 약 8500장

---

## 🔧 데이터 증강 전략

저조도 및 다양한 조명 환경에서의 인식률 향상을 위한 증강 기법을 적용했습니다.

### 1. 밝기 변화 증강 (5단계)

```python
# 다양한 조명 환경 시뮬레이션
brightness_levels = {
    'original': None,           # 원본
    'dark': 0.6,               # 어두운 환경
    'very_dark': 0.4,          # 매우 어두운 환경
    'bright': 1.3,             # 밝은 환경
    'low_contrast': 'custom'   # 저대비
}
```

|  증강 타입   | Alpha 값 | 설명                |
| :----------: | :------: | :------------------ |
|   Original   |   1.0    | 원본 이미지         |
|     Dark     |   0.6    | 어두운 실내 환경    |
|  Very Dark   |   0.4    | 야간/저조도 환경    |
|    Bright    |   1.3    | 밝은 야외 환경      |
| Low Contrast |  Custom  | 흐린 날씨/간접 조명 |

### 2. 노이즈 추가

```python
# 저조도 환경 시뮬레이션을 위한 가우시안 노이즈
gaussian_noise = np.random.normal(0, 25, image.shape)
noisy_image = np.clip(image + gaussian_noise, 0, 255)
```

### 3. 기하학적 변환

- **회전**: ±15도 범위 내 랜덤 회전
- **좌우 반전**: 50% 확률로 적용
- **스케일 변화**: 0.8~1.2 배율 조정
- **이동**: ±10% 범위 내 평행 이동

### 증강 효과

- **원본 데이터**: 2,000장
- **증강 후**: 8,500장 (4배 증가)
- **클래스 균형**: 각 카테고리별 동일한 증강 적용

---

## 📁 프로젝트 구조

```
SSbry/
├── dataset/                      # 학습 데이터셋
│   ├── images/
│   │   ├── train/               # 훈련 이미지 (80%)
│   │   │   ├── can_001.jpg
│   │   │   ├── glass_001.jpg
│   │   │   └── ...
│   │   └── val/                 # 검증 이미지 (20%)
│   │       ├── can_test_001.jpg
│   │       └── ...
│   └── labels/
│       ├── train/               # 훈련 라벨 (YOLO 형식)
│       │   ├── can_001.txt
│       │   └── ...
│       └── val/                 # 검증 라벨
│           ├── can_test_001.txt
│           └── ...
│
├── trashnet/                    # 원본 데이터 (증강 전)
│   ├── can/
│   ├── glass/
│   ├── paper/
│   ├── plastic/
│   ├── trash/
│   └── vinyl/
│
├── runs/                        # 학습 결과
│   └── detect/
│       └── waste_classification/
│           ├── weights/
│           │   ├── best.pt      # 최고 성능 모델
│           │   └── last.pt      # 마지막 체크포인트
│           ├── confusion_matrix.png
│           ├── results.png      # 학습 곡선
│           └── val_batch0_pred.jpg
│
├── best_waste_model.pt          # 최종 배포 모델
├── yolov8n_trash.onnx           # ONNX 변환 모델
├── dataset.yaml                 # 데이터셋 설정 파일
├── yolov8n.pt                   # YOLOv8 사전학습 모델
├── main.ipynb                   # 전체 실행 노트북
├── train.py                     # 학습 스크립트
├── export_onnx.py               # ONNX 변환 스크립트
└── requirements.txt             # Python 의존성
```

---

## 🎯 기술 스택

### Machine Learning

[![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-00FFFF?style=flat)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)

### Data Processing

[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-10.x-green?style=flat)](https://pillow.readthedocs.io/)

### Development

[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)

---

## 🚀 개발 환경 및 학습 과정

### 1️⃣ 환경 설정

**필수 패키지 설치**

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

**requirements.txt**

```txt
torch>=2.0.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
matplotlib>=3.7.0
```

**하드웨어 요구사항**

- **최소**: CPU (Intel i5 이상), RAM 8GB
- **권장**: GPU (CUDA 지원), RAM 16GB
- **저장공간**: 5GB 이상

### 2️⃣ 데이터 전처리

**자동 객체 탐지 및 라벨링**

```python
import cv2
import numpy as np

def auto_detect_object(image_path):
    """OpenCV 기반 객체 영역 자동 추출"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이진화 및 모폴로지 연산
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Contour 탐지
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 객체 선택
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return x, y, w, h
```

**YOLO 형식 라벨 생성**

```python
def convert_to_yolo_format(x, y, w, h, img_width, img_height, class_id):
    """바운딩 박스를 YOLO 형식으로 변환"""
    center_x = (x + w/2) / img_width
    center_y = (y + h/2) / img_height
    width = w / img_width
    height = h / img_height

    return f"{class_id} {center_x} {center_y} {width} {height}\n"
```

**데이터 분할**

- 훈련 데이터: 80% (2,000장 → 증강 후 10,000장)
- 검증 데이터: 20% (500장 → 증강 후 2,500장)

### 3️⃣ 모델 학습

**학습 파라미터**

```python
from ultralytics import YOLO

# YOLOv8n 사전학습 모델 로드
model = YOLO('yolov8n.pt')

# 학습 실행
results = model.train(
    data='dataset.yaml',        # 데이터셋 설정
    epochs=30,                  # 학습 횟수
    imgsz=416,                  # 입력 이미지 크기
    batch=4,                    # 배치 크기 (메모리 효율)
    device='cpu',               # 'cuda' for GPU
    patience=10,                # Early stopping
    save=True,                  # 체크포인트 저장
    project='runs/detect',      # 결과 저장 경로
    name='waste_classification',
    exist_ok=True
)
```

**dataset.yaml 설정**

```yaml
# 데이터셋 경로
path: ./dataset
train: images/train
val: images/val

# 클래스 정의
nc: 6 # number of classes
names: ["can", "glass", "paper", "plastic", "trash", "vinyl"]
```

**학습 시간**

- CPU: 약 1~2시간
- GPU (CUDA): 약 15~30분

### 4️⃣ 모델 평가 및 변환

**성능 평가**

```python
# 검증 데이터로 평가
metrics = model.val()

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

**ONNX 변환**

```python
# Flutter 앱 배포를 위한 ONNX 변환
model = YOLO('best_waste_model.pt')
model.export(
    format='onnx',
    opset=17,                # Flutter ONNX Runtime 호환
    simplify=True,
    dynamic=False,
    imgsz=416
)
```

---

## 💻 사용 방법

### Jupyter Notebook 실행

**main.ipynb**

```python
# 1단계: 설치 확인
!pip list | grep ultralytics

# 2단계: 데이터 준비 (증강 포함)
from data_augmentation import augment_dataset
augment_dataset('trashnet/', 'dataset/')

# 3단계: 모델 훈련
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=30)

# 4단계: 테스트 및 평가
model = YOLO('runs/detect/waste_classification/weights/best.pt')
results = model.predict('test_image.jpg', save=True)

# 5단계: ONNX 변환
model.export(format='onnx', opset=17)
```

### 명령줄 실행

```bash
# 학습
python train.py --data dataset.yaml --epochs 30 --imgsz 416

# 추론
python predict.py --weights best_waste_model.pt --source test_image.jpg

# ONNX 변환
python export_onnx.py --weights best_waste_model.pt --opset 17
```

### 결과 확인

학습 완료 후 `runs/detect/waste_classification/`에서 다음 결과 확인:

- **weights/best.pt**: 최고 성능 모델
- **confusion_matrix.png**: 혼동 행렬
- **results.png**: 학습 곡선 (Loss, mAP)
- **val_batch0_pred.jpg**: 검증 결과 시각화

---

## 📈 주요 특징

### 🎯 스마트 객체 탐지

- **OpenCV 기반 자동 객체 영역 추출**: Contour Detection으로 배경과 객체 분리
- **배경 노이즈 제거**: 모폴로지 연산으로 정확도 향상
- **자동 라벨링**: 수동 작업 최소화

### 🌓 다양한 환경 대응

- **5단계 밝기 변화 시뮬레이션**: 실내외 모든 조명 조건 커버
- **저조도 환경 노이즈 추가**: 야간/어두운 환경 대응
- **실제 사용 환경 고려**: 사용자가 촬영하는 다양한 각도와 거리 반영

### 📱 모바일 최적화

- **YOLOv8n (Nano) 모델**: 파라미터 수 최소화 (3.2M)
- **모델 크기**: 6.4MB (ONNX 형식)
- **추론 속도**: 모바일 CPU에서 평균 200ms
- **ONNX Runtime 호환**: Opset 17로 Flutter 완벽 연동

### 🔍 모델 성능

|     메트릭      |    값     |
| :-------------: | :-------: |
|      mAP50      | [값 입력] |
|    mAP50-95     | [값 입력] |
|    Precision    | [값 입력] |
|     Recall      | [값 입력] |
| 추론 속도 (CPU) |  ~200ms   |
|    모델 크기    |   6.4MB   |

---

## 🛠️ 트러블슈팅

### 1. ONNX Opset 버전 불일치

**문제**: Flutter ONNX Runtime이 Opset 18을 지원하지 않음

```python
# ✗ 기본 설정 (Opset 18)
model.export(format='onnx')

# ✓ 해결 방법
model.export(format='onnx', opset=17)
```

### 2. 메모리 부족 오류

**문제**: 배치 크기가 너무 커서 학습 중 메모리 부족

```python
# ✗ 큰 배치 크기
model.train(data='dataset.yaml', batch=16)

# ✓ 해결 방법
model.train(data='dataset.yaml', batch=4)
```

### 3. 클래스 불균형 문제

**문제**: 특정 카테고리의 인식률이 낮음

**해결 방법**:

- 부족한 클래스에 대한 추가 데이터 수집
- 증강 비율 조정으로 클래스 균형 맞추기
- Class Weights 적용

```python
# 클래스 가중치 적용
model.train(
    data='dataset.yaml',
    cls_weight=1.5  # 분류 손실 가중치 증가
)
```

---

## 🚀 향후 개선 계획

### 단기 계획

- [ ] 추가 쓰레기 카테고리 확장 (음식물, 의류 등)
- [ ] 모델 양자화를 통한 추가 경량화 (INT8)
- [ ] 재활용 가능/불가능 세부 분류

### 중기 계획

- [ ] 실시간 카메라 촬영 모드 지원
- [ ] 다국어 라벨링 데이터셋 구축
- [ ] 지역별 분리배출 규정 반영

### 장기 계획

- [ ] Transformer 기반 모델 실험 (DETR)
- [ ] Edge TPU 최적화
- [ ] 클라우드 기반 지속 학습 시스템

---

## 📊 데이터셋 출처

- **TrashNet Dataset** (Kaggle): 기본 이미지 데이터
- **자체 수집 데이터**: 한국 환경에 맞는 추가 데이터 200장
- **데이터 라이센스**: CC BY-NC 4.0

---

## 📝 참고사항

### 학습 권장 사항

- **학습 시간**: 약 30분~1시간 (CPU 기준)
- **최소 데이터**: 클래스당 200장 이상
- **Epoch**: 30번 이상 권장 (Early Stopping 적용)
- **권장 사양**: RAM 8GB 이상, GPU 권장

### 데이터 증강 팁

- 밝기 증강은 실제 사용 환경 반영 필수
- 과도한 증강은 오히려 성능 저하 가능
- 원본 데이터 품질이 가장 중요

### 모델 배포 시 주의사항

- ONNX Opset 버전 반드시 확인
- 입력 이미지 크기 일관성 유지 (416x416)
- 전처리 파이프라인 동일하게 적용

---

<div align="center">

**AI로 만드는 지속 가능한 미래** 🌍♻️

</div>
