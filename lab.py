# ============================================================================
# Quantum Circuit-based Aircraft Data Classifier
# ============================================================================
# 목적: 항공기 센서 데이터를 활용하여 이륙 가능 여부를 분류
# 데이터셋: NASA Turbofan Engine Degradation Simulation Dataset
# 모델: 고전 SVM vs 양자 분류기 (QSVM, QNN)
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pennylane as qml
from pennylane import numpy as pnp
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("Quantum Circuit-based Aircraft Data Classifier")
print("=" * 80)


# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================
class AircraftDataLoader:
    """
    NASA Turbofan 데이터셋 로드 및 전처리
    - 데이터 로드
    - 결측값 처리
    - 특성 선택 (고도, 속도, 기온, 고장 플래그)
    - 이륙 가능 여부 레이블 생성
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self, n_samples=500):
        """
        실제 데이터셋이 없는 경우 합성 데이터 생성
        항공기 센서 데이터 시뮬레이션
        """
        print("\n[1단계] 합성 항공기 센서 데이터 생성 중...")
        
        np.random.seed(42)
        
        # 특성 생성 (정규 분포 기반)
        altitude = np.random.normal(1000, 300, n_samples)  # 고도 (m)
        speed = np.random.normal(250, 50, n_samples)       # 속도 (km/h)
        temperature = np.random.normal(15, 10, n_samples)  # 기온 (°C)
        fault_flag = np.random.binomial(1, 0.2, n_samples) # 고장 플래그 (0 or 1)
        
        # 이륙 가능 여부 결정 로직
        # 조건: 고도 > 800, 속도 > 200, 기온 > 0, 고장 없음
        takeoff_possible = (
            (altitude > 800) & 
            (speed > 200) & 
            (temperature > 0) & 
            (fault_flag == 0)
        ).astype(int)
        
        # DataFrame 생성
        data = pd.DataFrame({
            'altitude': altitude,
            'speed': speed,
            'temperature': temperature,
            'fault_flag': fault_flag,
            'takeoff_possible': takeoff_possible
        })
        
        print(f"✓ 생성된 데이터 샘플 수: {n_samples}")
        print(f"✓ 이륙 가능: {takeoff_possible.sum()} ({takeoff_possible.mean()*100:.1f}%)")
        print(f"✓ 이륙 불가능: {(1-takeoff_possible).sum()} ({(1-takeoff_possible.mean())*100:.1f}%)")
        
        return data
    
    def load_and_preprocess(self, n_samples=500):
        """
        데이터 로드 및 전처리 파이프라인
        """
        # 합성 데이터 생성 (실제 NASA 데이터셋 사용 시 수정 필요)
        data = self.generate_synthetic_data(n_samples)
        
        # 특성과 레이블 분리
        X = data[['altitude', 'speed', 'temperature', 'fault_flag']].values
        y = data['takeoff_possible'].values
        
        # 학습/테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 정규화
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n✓ 학습 데이터: {X_train_scaled.shape[0]} 샘플")
        print(f"✓ 테스트 데이터: {X_test_scaled.shape[0]} 샘플")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, data


# ============================================================================
# 2. 고전 SVM 분류기
# ============================================================================
class ClassicalSVM:
    """
    고전적인 Support Vector Machine 분류기
    """
    
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVC(kernel=kernel, C=C, random_state=42)
        self.kernel = kernel
        
    def train(self, X_train, y_train):
        """SVM 모델 학습"""
        print(f"\n[2단계] 고전 SVM 학습 중 (커널: {self.kernel})...")
        self.model.fit(X_train, y_train)
        print("✓ 학습 완료")
        
    def predict(self, X):
        """예측 수행"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """모델 평가"""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n[고전 SVM 결과]")
        print(f"정확도: {accuracy*100:.2f}%")
        print("\n분류 리포트:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['이륙 불가능', '이륙 가능']))
        
        return accuracy, y_pred


# ============================================================================
# 3. 양자 회로 설계 (Angle Encoding)
# ============================================================================
class QuantumCircuit:
    """
    PennyLane을 활용한 양자 회로 설계
    - Angle Encoding: 입력 데이터를 회전 게이트의 각도로 인코딩
    - Variational Circuit: 학습 가능한 파라미터 포함
    """
    
    def __init__(self, n_qubits=4, n_layers=2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
    def angle_encoding(self, x):
        """
        Angle Encoding: 데이터를 RY 회전 게이트로 인코딩
        """
        for i in range(min(len(x), self.n_qubits)):
            qml.RY(x[i], wires=i)
    
    def variational_layer(self, weights):
        """
        Variational Layer: 학습 가능한 파라미터 층
        """
        for i in range(self.n_qubits):
            qml.RY(weights[i, 0], wires=i)
            qml.RZ(weights[i, 1], wires=i)
        
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_qubits - 1, 0])
    
    def create_circuit(self):
        """
        양자 회로 생성
        """
        @qml.qnode(self.dev)
        def circuit(weights, x):
            # 데이터 인코딩
            self.angle_encoding(x)
            
            # Variational layers
            for i in range(self.n_layers):
                self.variational_layer(weights[i])
            
            # 측정: Pauli-Z 기댓값
            return qml.expval(qml.PauliZ(0))
        
        return circuit


# ============================================================================
# 4. Quantum SVM (QSVM)
# ============================================================================
class QuantumSVM:
    """
    Quantum Kernel을 사용한 SVM
    """
    
    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.model = None
        
    def quantum_kernel(self, x1, x2):
        """
        양자 커널 함수: 두 데이터 포인트 간의 유사도 계산
        """
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            # x1 인코딩
            for i in range(min(len(x1), self.n_qubits)):
                qml.RY(x1[i], wires=i)
            
            # x2 인코딩 (역방향)
            for i in range(min(len(x2), self.n_qubits)):
                qml.RY(-x2[i], wires=i)
            
            # 측정
            return qml.probs(wires=range(self.n_qubits))
        
        probs = kernel_circuit(x1, x2)
        # |<0|0>|^2 = 첫 번째 확률
        return probs[0]
    
    def compute_kernel_matrix(self, X1, X2):
        """
        커널 행렬 계산
        """
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.quantum_kernel(X1[i], X2[j])
        
        return K
    
    def train(self, X_train, y_train):
        """
        QSVM 학습
        """
        print(f"\n[4단계] Quantum SVM 학습 중...")
        print("커널 행렬 계산 중... (시간이 소요될 수 있습니다)")
        
        # 학습 데이터의 커널 행렬 계산
        K_train = self.compute_kernel_matrix(X_train, X_train)
        
        # SVM 학습 (precomputed kernel)
        self.model = SVC(kernel='precomputed', random_state=42)
        self.model.fit(K_train, y_train)
        
        self.X_train = X_train
        print("✓ QSVM 학습 완료")
        
    def predict(self, X_test):
        """
        예측 수행
        """
        # 테스트 데이터와 학습 데이터 간의 커널 행렬 계산
        K_test = self.compute_kernel_matrix(X_test, self.X_train)
        return self.model.predict(K_test)
    
    def evaluate(self, X_test, y_test):
        """
        모델 평가
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n[Quantum SVM 결과]")
        print(f"정확도: {accuracy*100:.2f}%")
        print("\n분류 리포트:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['이륙 불가능', '이륙 가능']))
        
        return accuracy, y_pred


# ============================================================================
# 5. Quantum Neural Network (QNN)
# ============================================================================
class QuantumNeuralNetwork:
    """
    Variational Quantum Classifier (VQC) 기반 QNN
    """
    
    def __init__(self, n_qubits=4, n_layers=2, learning_rate=0.01):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        
        # 양자 회로 생성
        qc = QuantumCircuit(n_qubits, n_layers)
        self.circuit = qc.create_circuit()
        
        # 파라미터 초기화
        self.weights = pnp.random.randn(n_layers, n_qubits, 2, requires_grad=True)
        
    def predict_single(self, x):
        """
        단일 샘플 예측
        """
        output = self.circuit(self.weights, x)
        return 1 if output > 0 else 0
    
    def predict(self, X):
        """
        다중 샘플 예측
        """
        return np.array([self.predict_single(x) for x in X])
    
    def cost_function(self, weights, X, y):
        """
        손실 함수: 평균 제곱 오차
        """
        predictions = []
        for x in X:
            output = self.circuit(weights, x)
            predictions.append(output)
        
        predictions = pnp.array(predictions)
        # 레이블을 -1, 1로 변환
        y_transformed = 2 * y - 1
        
        return pnp.mean((predictions - y_transformed) ** 2)
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """
        QNN 학습 (경사하강법)
        """
        print(f"\n[5단계] Quantum Neural Network 학습 중...")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}")
        
        opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        
        n_samples = len(X_train)
        costs = []
        
        for epoch in range(epochs):
            # 배치 단위 학습
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_cost = 0
            n_batches = n_samples // batch_size
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # 그래디언트 업데이트
                self.weights, cost = opt.step_and_cost(
                    lambda w: self.cost_function(w, X_batch, y_batch),
                    self.weights
                )
                
                epoch_cost += cost
            
            avg_cost = epoch_cost / n_batches
            costs.append(avg_cost)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Cost: {avg_cost:.4f}")
        
        print("✓ QNN 학습 완료")
        return costs
    
    def evaluate(self, X_test, y_test):
        """
        모델 평가
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n[Quantum Neural Network 결과]")
        print(f"정확도: {accuracy*100:.2f}%")
        print("\n분류 리포트:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['이륙 불가능', '이륙 가능']))
        
        return accuracy, y_pred


# ============================================================================
# 6. 결과 비교 및 시각화
# ============================================================================
class ResultVisualizer:
    """
    모델 성능 비교 및 시각화
    """
    
    @staticmethod
    def plot_confusion_matrices(y_test, predictions_dict):
        """
        혼동 행렬 시각화
        """
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name} 혼동 행렬')
            axes[idx].set_ylabel('실제 값')
            axes[idx].set_xlabel('예측 값')
            axes[idx].set_xticklabels(['불가능', '가능'])
            axes[idx].set_yticklabels(['불가능', '가능'])
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("\n✓ 혼동 행렬 저장: confusion_matrices.png")
        plt.show()
    
    @staticmethod
    def plot_accuracy_comparison(accuracies_dict):
        """
        정확도 비교 막대 그래프
        """
        models = list(accuracies_dict.keys())
        accuracies = list(accuracies_dict.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
        
        # 막대 위에 정확도 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height*100:.2f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.ylim(0, 1.1)
        plt.ylabel('정확도', fontsize=12)
        plt.title('모델별 분류 정확도 비교', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ 정확도 비교 그래프 저장: accuracy_comparison.png")
        plt.show()
    
    @staticmethod
    def plot_data_distribution(data):
        """
        데이터 분포 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        features = ['altitude', 'speed', 'temperature', 'fault_flag']
        feature_names = ['고도 (m)', '속도 (km/h)', '기온 (°C)', '고장 플래그']
        
        for idx, (feature, name) in enumerate(zip(features, feature_names)):
            ax = axes[idx // 2, idx % 2]
            
            data_0 = data[data['takeoff_possible'] == 0][feature]
            data_1 = data[data['takeoff_possible'] == 1][feature]
            
            ax.hist(data_0, alpha=0.5, label='이륙 불가능', bins=20, color='red')
            ax.hist(data_1, alpha=0.5, label='이륙 가능', bins=20, color='green')
            
            ax.set_xlabel(name, fontsize=11)
            ax.set_ylabel('빈도', fontsize=11)
            ax.set_title(f'{name} 분포', fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ 데이터 분포 그래프 저장: data_distribution.png")
        plt.show()


# ============================================================================
# 7. 메인 실행 파이프라인
# ============================================================================
def main():
    """
    전체 파이프라인 실행
    """
    print("\n" + "="*80)
    print("항공기 이륙 가능 여부 분류 시스템")
    print("고전 SVM vs 양자 분류기 (QSVM, QNN) 비교")
    print("="*80 + "\n")
    
    # 1. 데이터 로드 및 전처리
    loader = AircraftDataLoader()
    X_train, X_test, y_train, y_test, data = loader.load_and_preprocess(n_samples=500)
    
    # 데이터 분포 시각화
    visualizer = ResultVisualizer()
    visualizer.plot_data_distribution(data)
    
    # 2. 고전 SVM
    classical_svm = ClassicalSVM(kernel='rbf', C=1.0)
    classical_svm.train(X_train, y_train)
    acc_classical, pred_classical = classical_svm.evaluate(X_test, y_test)
    
    # 3. Quantum SVM (샘플 축소 - 계산 시간 단축)
    # 전체 데이터 사용 시 시간이 매우 오래 걸릴 수 있음
    sample_size = min(100, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_train_sample = X_train[indices]
    y_train_sample = y_train[indices]
    
    qsvm = QuantumSVM(n_qubits=4)
    qsvm.train(X_train_sample, y_train_sample)
    acc_qsvm, pred_qsvm = qsvm.evaluate(X_test, y_test)
    
    # 4. Quantum Neural Network
    qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2, learning_rate=0.01)
    costs = qnn.train(X_train, y_train, epochs=30, batch_size=32)
    acc_qnn, pred_qnn = qnn.evaluate(X_test, y_test)
    
    # 5. 결과 비교 및 시각화
    print("\n" + "="*80)
    print("최종 결과 요약")
    print("="*80)
    
    accuracies = {
        '고전 SVM': acc_classical,
        'Quantum SVM': acc_qsvm,
        'Quantum NN': acc_qnn
    }
    
    predictions = {
        '고전 SVM': pred_classical,
        'Quantum SVM': pred_qsvm,
        'Quantum NN': pred_qnn
    }
    
    print("\n[모델별 정확도]")
    for model, acc in accuracies.items():
        print(f"  {model}: {acc*100:.2f}%")
    
    # 시각화
    visualizer.plot_accuracy_comparison(accuracies)
    visualizer.plot_confusion_matrices(y_test, predictions)
    
    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80)
    
    return accuracies, predictions


# ============================================================================
# 실행
# ============================================================================
if __name__ == "__main__":
    main()
