"""
Quantum Support Vector Machine (QSVM)
양자 커널을 사용한 SVM
"""

import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


class QuantumSVM:
    """
    양자 커널 기반 SVM
    """
    
    def __init__(self, config=None):
        """
        Args:
            config (dict, optional): 모델 설정
        """
        if config is None:
            config = {'n_qubits': 4, 'sample_size': 100}
        
        self.n_qubits = config.get('n_qubits', 4)
        self.sample_size = config.get('sample_size', 100)
        
        # 양자 디바이스 초기화
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        
        self.model = None
        self.X_train = None
        self.is_trained = False
        
    def quantum_kernel(self, x1, x2):
        """
        양자 커널 함수
        두 데이터 포인트 간의 유사도를 양자 상태로 계산
        
        Args:
            x1, x2: 입력 벡터
            
        Returns:
            float: 커널 값
        """
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            # x1 인코딩 (Angle Encoding)
            for i in range(min(len(x1), self.n_qubits)):
                qml.RY(x1[i], wires=i)
            
            # x2 역인코딩
            for i in range(min(len(x2), self.n_qubits)):
                qml.RY(-x2[i], wires=i)
            
            # 측정: 모든 큐비트가 |0> 상태일 확률
            return qml.probs(wires=range(self.n_qubits))
        
        probs = kernel_circuit(x1, x2)
        return probs[0]  # |00...0> 확률
    
    def compute_kernel_matrix(self, X1, X2):
        """
        커널 행렬 계산
        
        Args:
            X1, X2: 데이터 행렬
            
        Returns:
            np.ndarray: 커널 행렬
        """
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        total = n1 * n2
        count = 0
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.quantum_kernel(X1[i], X2[j])
                count += 1
                
                # 진행률 표시 (10% 단위)
                if count % max(1, total // 10) == 0:
                    print(f"  커널 계산 진행률: {count}/{total} ({count/total*100:.0f}%)")
        
        return K
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        QSVM 학습
        계산 시간 단축을 위해 샘플링 가능
        
        Args:
            X_train, y_train: 학습 데이터
            X_val, y_val: 검증 데이터 (옵션)
        """
        print("\n" + "="*60)
        print("Quantum SVM 학습 시작")
        print("="*60)
        
        # 샘플 크기 제한 (계산 시간 단축)
        if len(X_train) > self.sample_size:
            print(f"⚠ 계산 시간 단축을 위해 {self.sample_size}개 샘플로 제한")
            indices = np.random.choice(len(X_train), self.sample_size, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
        
        print(f"학습 샘플 수: {len(X_train)}")
        print("양자 커널 행렬 계산 중... (시간이 소요됩니다)")
        
        # 학습 데이터 커널 행렬 계산
        K_train = self.compute_kernel_matrix(X_train, X_train)
        
        print("\nSVM 학습 중...")
        # Precomputed 커널로 SVM 학습
        self.model = SVC(kernel='precomputed', random_state=42)
        self.model.fit(K_train, y_train)
        
        self.X_train = X_train
        self.is_trained = True
        
        # 학습 데이터 정확도
        train_acc = accuracy_score(y_train, self.model.predict(K_train))
        print(f"✓ 학습 데이터 정확도: {train_acc*100:.2f}%")
        
        # 검증 데이터 평가
        if X_val is not None and y_val is not None:
            val_acc = self.evaluate(X_val, y_val, verbose=False)
            print(f"✓ 검증 데이터 정확도: {val_acc*100:.2f}%")
        
        print("✓ QSVM 학습 완료\n")
    
    def predict(self, X_test):
        """
        예측 수행
        
        Args:
            X_test: 테스트 데이터
            
        Returns:
            np.ndarray: 예측 레이블
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        print("양자 커널 계산 중 (테스트 데이터)...")
        K_test = self.compute_kernel_matrix(X_test, self.X_train)
        
        return self.model.predict(K_test)
    
    def evaluate(self, X, y, verbose=True):
        """
        모델 평가
        
        Args:
            X, y: 테스트 데이터
            verbose (bool): 상세 출력
            
        Returns:
            float: 정확도
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        if verbose:
            print("\n" + "="*60)
            print("Quantum SVM Evaluation Results")
            print("="*60)
            print(f"Accuracy: {accuracy*100:.2f}%\n")
            print("Classification Report:")
            print(classification_report(y, y_pred,
                                       target_names=['Not Possible', 'Possible']))
        
        return accuracy
    
    def save_model(self, save_path='./checkpoints/qsvm.pkl'):
        """모델 저장"""
        if not self.is_trained:
            raise ValueError("학습된 모델이 없습니다.")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model_data = {
            'model': self.model,
            'X_train': self.X_train,
            'n_qubits': self.n_qubits
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ QSVM 모델 저장: {save_path}")
    
    def load_model(self, load_path='./checkpoints/qsvm.pkl'):
        """저장된 모델 로드"""
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.X_train = model_data['X_train']
        self.n_qubits = model_data['n_qubits']
        self.is_trained = True
        
        print(f"✓ QSVM 모델 로드: {load_path}")
