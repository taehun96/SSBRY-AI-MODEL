# SSbry - AI 학습 모델

## 🤖 YOLOv8 기반 스마트 분리배출 AI 모델
### 📌 모델 개요
실시간 쓰레기 탐지 및 6종 분류를 위한 YOLOv8 커스텀 모델
**왜 YOLOv8을 선택했나?**
- 실시간 탐지: 모바일 환경에서도 빠른 추론 속도 (YOLOv8n 경량화 모델)
- 높은 정확도: 작은 모델 크기 대비 우수한 분류 성능
- 다중 객체 탐지: 한 이미지에서 여러 쓰레기 동시 분류 가능
- 쉬운 배포: Flutter 앱과의 원활한 연동

### 🗂️ 데이터셋 구성
**분류 카테고리 (6종)**
- 🥫 캔 (can) - 알루미늄 캔류
- 🍾 유리 (glass) - 유리병류
- 📄 종이 (paper) - 종이류
- ♻️ 플라스틱 (plastic) - 플라스틱류
- 🗑️ 일반쓰레기 (trash) - 일반 폐기물
- 📦 비닐 (vinyl) - 비닐류

### 데이터 증강 전략
저조도 및 다양한 조명 환경에서의 인식률 향상을 위한 증강 기법 적용:
**밝기 변화 증강 (5단계)**
- Original (원본)
- Dark (어두운 환경, alpha=0.6)  
- Very Dark (매우 어두운 환경, alpha=0.4)
- Bright (밝은 환경, alpha=1.3)
- Low Contrast (저대비)

**노이즈 추가**
- 저조도 환경 시뮬레이션을 위한 가우시안 노이즈

### 🔧 개발 환경 및 학습 과정
**1단계: 환경 설정**
필수 패키지
- PyTorch (CPU/GPU 자동 감지)
- Ultralytics YOLOv8
- OpenCV
- NumPy

**2단계: 데이터 전처리**
- 자동 객체 탐지: OpenCV 기반 contour detection으로 객체 영역 자동 추출
- YOLO 형식 라벨링: 정규화된 좌표 (center_x, center_y, width, height)
- 데이터 분할: 훈련 80% / 검증 20%

**3단계: 모델 학습**
학습 파라미터
- model = YOLO('yolov8n.pt')
```
training_params = {
    'epochs': 30,            # 학습 횟수
    'imgsz': 416,           # 입력 이미지 크기
    'batch': 4,             # 배치 크기 (메모리 효율)
    'patience': 10,         # Early stopping
    'device': 'cpu'         # CPU 학습 (GPU 미사용시)
}
```
**4단계: 학습 기반 최종 파일 생성**
- best_waste_model.pt생성성

### 📊 AI 학습 모델델 구조
```
📁 SSbry/
├── 📁 dataset/                    # 학습 데이터셋
│   ├── 📁 images/
│   │   ├── 📁 train/             # 훈련 이미지
│   │   └── 📁 val/               # 검증 이미지
│   └── 📁 labels/
│       ├── 📁 train/             # 훈련 라벨
│       └── 📁 val/               # 검증 라벨
├── 📁 runs/
│   └── 📁 detect/
│       └── 📁 waste_classification/  # 학습 결과
├── 📁 trashnet/                  # 원본 데이터
│   ├── 📁 can/
│   ├── 📁 glass/
│   ├── 📁 paper/
│   ├── 📁 plastic/
│   ├── 📁 trash/
│   └── 📁 vinyl/
├── 📄 best_waste_model.pt        # 최종 학습 모델
├── 📄 dataset.yaml               # 데이터셋 설정
├── 📄 yolov8n.pt                # 사전학습 모델
└── 📄 main.ipynb                # 전체 실행 노트북
```

### 💻 사용 방법
- main.ipynb
- 1단계 설치확인
- 2단계 데이터 준비(증강 포함)
- 3단계 모델 훈련
- 4단계 테스트 및 평가(저장된 이미지 분석)
- runs/detect/waste_classfification/에서 학습 결과 확인
- best_waste_model.pt가 생성되어 있는지 확인

### 📈 주요 특징
**🎯 스마트 객체 탐지**
- OpenCV 기반 자동 객체 영역 추출
- 배경 노이즈 제거를 위한 모폴로지 연산

**🌓 다양한 환경 대응**
- 5단계 밝기 변화 시뮬레이션
- 저조도 환경 노이즈 추가
- 실제 사용 환경을 고려한 데이터 증강

**📱 모바일 최적화**
- YOLOv8n (Nano) 모델 사용으로 경량화

**🚀 향후 개선 계획**
- 추가 쓰레기 카테고리 확장 (음식물, 의류 등)
- 모델 양자화를 통한 추가 경량화
- 재활용 가능/불가능 세부 분류

**📝 참고사항**
- 학습 시간: 약 30분~1시간 (CPU 기준)
- 최소 데이터: 클래스당 20장 이상
- Epoch(학습 횟수): 30번 이상 권장장
- 권장 사양: 8GB RAM 이상
