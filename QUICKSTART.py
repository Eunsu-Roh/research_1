"""
빠른 시작 가이드
"""

# ============================================================================
# 항공기 이륙 가능 여부 분류 - 양자 머신러닝 프로젝트
# ============================================================================

print("""
╔══════════════════════════════════════════════════════════════════════╗
║          항공기 이륙 가능 여부 분류 시스템                            ║
║          고전 ML vs 양자 ML 비교 연구                                 ║
╚══════════════════════════════════════════════════════════════════════╝

📁 프로젝트 구조 (PyTorch 스타일)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

연구발표/
├── configs/          - 설정 파일 (하이퍼파라미터)
├── data/             - 데이터 전처리 모듈
│   ├── preprocessing.py   - 데이터 로드, 정규화, 분할
│   ├── dataset.py         - PyTorch 스타일 Dataset
│   └── processed/         - 전처리된 Train/Val/Test 데이터
├── models/           - 모델 정의
│   ├── classical_svm.py   - 고전 SVM
│   ├── qsvm.py            - Quantum SVM
│   └── qnn.py             - Quantum Neural Network
├── utils/            - 유틸리티
│   ├── visualization.py   - 그래프 생성
│   └── metrics.py         - 평가 지표
├── checkpoints/      - 학습된 모델 (.pkl)
├── results/          - 실험 결과 (그래프, 리포트)
├── main.py           - 전체 파이프라인 실행
├── train.py          - 학습만 실행
└── evaluate.py       - 평가만 실행


🚀 실행 방법
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  전체 파이프라인 실행 (추천)
   python main.py
   
   → 데이터 생성 → 학습 → 평가 → 결과 저장
   
2️⃣  학습만 실행
   python train.py
   
   → 3개 모델 학습 후 체크포인트 저장
   
3️⃣  평가만 실행 (학습된 모델 필요)
   python evaluate.py
   
   → 저장된 모델 로드 후 테스트 데이터로 평가


📊 출력 결과
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

results/
├── data_distribution.png     - 특성별 데이터 분포
├── confusion_matrices.png    - 3개 모델 혼동 행렬
├── accuracy_comparison.png   - 정확도 비교 막대 그래프
├── training_history.png      - QNN 학습 과정 (Loss/Acc)
└── summary_report.txt        - 실험 결과 요약


🔬 실험 결과 (현재)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

모델              정확도      특징
────────────────────────────────────────────────────
고전 SVM          97.00%     최고 성능, RBF 커널
Quantum SVM       90.00%     양자 커널 기반
Quantum NN        64.00%     Variational Circuit


💡 주요 특징
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ PyTorch 스타일 프로젝트 구조
   - 모듈화된 코드 (data, models, utils 분리)
   - Dataset 클래스 (PyTorch 인터페이스)
   - 체크포인트 저장/로드
   - YAML 설정 파일

✅ 데이터 관리
   - Train/Validation/Test 분할
   - StandardScaler 정규화
   - .npy 형식으로 저장 (빠른 로드)

✅ 양자 머신러닝
   - Angle Encoding (데이터 → 양자 상태)
   - Variational Circuit (학습 가능한 양자 게이트)
   - 양자 커널 (SVM용)

✅ 실험 재현성
   - 랜덤 시드 고정
   - 설정 파일로 하이퍼파라미터 관리
   - 모든 결과 자동 저장


⚙️  하이퍼파라미터 수정
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

configs/config.yaml 수정:

data:
  n_samples: 500        # 데이터 샘플 수
  test_size: 0.2        # 테스트 비율
  val_size: 0.1         # 검증 비율

models:
  qnn:
    epochs: 50          # 학습 에포크 (↑하면 성능 향상 가능)
    batch_size: 32      # 배치 크기
    learning_rate: 0.01 # 학습률 (0.001~0.1 추천)
    n_layers: 2         # 양자 회로 층 수


🔧 커스텀 실험
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1. QNN만 학습하기
from models.qnn import QuantumNeuralNetwork
import yaml

with open('./configs/config.yaml') as f:
    config = yaml.safe_load(f)

qnn = QuantumNeuralNetwork(config['models']['qnn'])
qnn.train(X_train, y_train, X_val, y_val)
qnn.save_model()

# 2. 저장된 모델 로드
qnn.load_model('./checkpoints/qnn.pkl')
accuracy = qnn.evaluate(X_test, y_test)

# 3. 단일 예측
prediction = qnn.predict_single([1.2, -0.5, 0.3, 0])


📚 다음 단계
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NASA Turbofan 실제 데이터 연동
2. QNN 하이퍼파라미터 튜닝 (epochs ↑, learning_rate 조정)
3. 더 깊은 양자 회로 (n_layers 증가)
4. 다중 클래스 분류 (정상/경고/위험)
5. 앙상블 모델 (고전 + 양자)


📞 도움말
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- README.md 참조
- show_structure.py 실행으로 전체 구조 확인
- 각 .py 파일 상단에 주석으로 설명 있음

""")
