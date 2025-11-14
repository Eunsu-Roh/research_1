# Quantum Aircraft Classifier

항공기 센서 데이터를 활용한 이륙 가능 여부 분류 시스템  
**고전 머신러닝 vs 양자 머신러닝 비교 연구**

## 📁 프로젝트 구조

```
연구발표/
├── configs/              # 설정 파일
│   └── config.yaml      # 하이퍼파라미터 및 실험 설정
│
├── data/                # 데이터 관련 모듈
│   ├── __init__.py
│   ├── preprocessing.py # 데이터 전처리 (로드, 정규화, 분할)
│   ├── dataset.py       # PyTorch 스타일 Dataset 클래스
│   └── processed/       # 전처리된 데이터 저장 (자동 생성)
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       └── y_test.npy
│
├── models/              # 모델 정의
│   ├── __init__.py
│   ├── classical_svm.py # 고전 Support Vector Machine
│   ├── qsvm.py          # Quantum SVM (양자 커널)
│   └── qnn.py           # Quantum Neural Network
│
├── utils/               # 유틸리티 함수
│   ├── __init__.py
│   ├── visualization.py # 시각화 (그래프, 혼동 행렬)
│   └── metrics.py       # 평가 지표 계산
│
├── checkpoints/         # 학습된 모델 저장 (자동 생성)
│   ├── classical_svm.pkl
│   ├── qsvm.pkl
│   └── qnn.pkl
│
├── results/             # 실험 결과 (자동 생성)
│   ├── data_distribution.png
│   ├── confusion_matrices.png
│   ├── accuracy_comparison.png
│   ├── training_history.png
│   └── summary_report.txt
│
├── main.py              # 전체 파이프라인 실행
├── train.py             # 학습 스크립트
├── evaluate.py          # 평가 스크립트
├── lab.py               # 통합 버전 (이전 파일)
└── README.md            # 프로젝트 문서 (이 파일)
```

## 🚀 빠른 시작

### 1. 환경 설정

```powershell
conda activate env
pip install numpy pandas matplotlib seaborn scikit-learn pennylane pyyaml
```

### 2. 전체 파이프라인 실행

```powershell
python main.py
```

이 명령어는 다음을 자동으로 수행합니다:
- 데이터 생성 및 전처리
- Train/Validation/Test 분할
- 3개 모델 학습 (고전 SVM, QSVM, QNN)
- 모델 평가 및 비교
- 결과 시각화 및 저장

### 3. 개별 실행

#### 학습만 실행
```powershell
python train.py
```

#### 평가만 실행 (학습된 모델 필요)
```powershell
python evaluate.py
```

## 📊 데이터셋

### 특성 (Features)
- **고도 (Altitude)**: 항공기 고도 (m)
- **속도 (Speed)**: 항공기 속도 (km/h)
- **기온 (Temperature)**: 외부 기온 (°C)
- **고장 플래그 (Fault Flag)**: 시스템 고장 여부 (0/1)

### 레이블 (Label)
- **이륙 가능 (1)**: 안전한 이륙 조건
- **이륙 불가능 (0)**: 이륙 불가 조건

### 데이터 분할
- **Train**: 70% (학습용)
- **Validation**: 10% (검증용)
- **Test**: 20% (최종 평가용)

## 🤖 모델

### 1. 고전 SVM (Classical SVM)
- **커널**: RBF (Radial Basis Function)
- **목적**: 기준선(baseline) 성능 제공
- **구현**: Scikit-learn

### 2. Quantum SVM (QSVM)
- **양자 커널**: Angle Encoding 기반
- **큐비트 수**: 4
- **목적**: 양자 커널의 표현력 검증
- **구현**: PennyLane + Scikit-learn

### 3. Quantum Neural Network (QNN)
- **구조**: Variational Quantum Classifier
- **인코딩**: Angle Encoding
- **층 수**: 2 layers
- **최적화**: 경사하강법
- **구현**: PennyLane

## ⚙️ 설정 (config.yaml)

주요 설정 파라미터:

```yaml
data:
  n_samples: 500        # 생성할 데이터 샘플 수
  test_size: 0.2        # 테스트 데이터 비율
  val_size: 0.1         # 검증 데이터 비율

models:
  qnn:
    epochs: 50          # 학습 에포크
    batch_size: 32      # 배치 크기
    learning_rate: 0.01 # 학습률
```

## 📈 출력 결과

### 1. 콘솔 출력
- 데이터 생성 정보
- 학습 진행 상황
- 모델별 정확도 및 분류 리포트
- 최고 성능 모델

### 2. 저장 파일

#### 그래프 (`results/`)
- `data_distribution.png`: 특성별 데이터 분포
- `confusion_matrices.png`: 모델별 혼동 행렬
- `accuracy_comparison.png`: 정확도 비교
- `training_history.png`: QNN 학습 과정

#### 모델 체크포인트 (`checkpoints/`)
- `classical_svm.pkl`: 학습된 고전 SVM
- `qsvm.pkl`: 학습된 QSVM
- `qnn.pkl`: 학습된 QNN

#### 데이터 (`data/processed/`)
- 전처리된 Train/Val/Test 데이터
- Scaler 객체 (추론 시 재사용)

## 🔬 주요 특징

### PyTorch 스타일 구조
- **모듈화**: 데이터, 모델, 유틸리티 분리
- **Dataset 클래스**: PyTorch Dataset 인터페이스 구현
- **체크포인트**: 학습된 모델 저장/로드
- **재현성**: 시드 고정 및 설정 파일 관리

### 양자 머신러닝 구현
- **Angle Encoding**: 고전 데이터를 양자 상태로 변환
- **Variational Circuit**: 학습 가능한 파라미터 층
- **양자 커널**: 양자 유사도 기반 SVM
- **하이브리드**: 고전-양자 결합 구조

## 📝 사용 예시

### 커스텀 설정으로 실행
```python
# custom_config.yaml 생성 후
python main.py --config ./configs/custom_config.yaml
```

### 개별 모델 학습
```python
from models.qnn import QuantumNeuralNetwork
from data.preprocessing import AircraftDataPreprocessor

# 데이터 준비
config = load_config()
preprocessor = AircraftDataPreprocessor(config)
X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocessor.process_pipeline()

# QNN 학습
qnn = QuantumNeuralNetwork(config['models']['qnn'])
qnn.train(X_train, y_train, X_val, y_val)
qnn.save_model()

# 평가
accuracy = qnn.evaluate(X_test, y_test)
```

## 🎯 확장 아이디어

1. **실제 데이터 연동**: NASA Turbofan Dataset 사용
2. **다중 클래스 분류**: 정상/경고/위험 3단계 분류
3. **앙상블**: 양자-고전 하이브리드 앙상블
4. **최적화**: 하이퍼파라미터 튜닝
5. **강화학습**: 경로 판단 보조 시스템

## 📚 참고 자료

- **PennyLane**: https://pennylane.ai/
- **Qiskit**: https://qiskit.org/
- **NASA Dataset**: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- **Quantum Machine Learning**: https://arxiv.org/abs/1611.09347

## 📄 라이선스

본 프로젝트는 연구 및 교육 목적으로 제작되었습니다.

---

**작성자**: 학부연구생  
**날짜**: 2025년 11월 14일
