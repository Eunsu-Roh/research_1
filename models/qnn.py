"""
Quantum Neural Network (QNN)
Variational Quantum Classifier
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


class QuantumCircuit:
    """
    양자 회로 설계 (Angle Encoding + Variational Layer)
    """
    
    def __init__(self, n_qubits=4, n_layers=2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device('default.qubit', wires=n_qubits)
    
    def angle_encoding(self, x):
        """데이터를 RY 회전으로 인코딩"""
        for i in range(min(len(x), self.n_qubits)):
            qml.RY(x[i], wires=i)
    
    def variational_layer(self, weights):
        """학습 가능한 파라미터 층"""
        for i in range(self.n_qubits):
            qml.RY(weights[i, 0], wires=i)
            qml.RZ(weights[i, 1], wires=i)
        
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[self.n_qubits - 1, 0])
    
    def create_circuit(self):
        """양자 회로 생성"""
        @qml.qnode(self.dev)
        def circuit(weights, x):
            self.angle_encoding(x)
            
            for i in range(self.n_layers):
                self.variational_layer(weights[i])
            
            return qml.expval(qml.PauliZ(0))
        
        return circuit


class QuantumNeuralNetwork:
    """
    Variational Quantum Classifier (VQC) 기반 QNN
    """
    
    def __init__(self, config=None):
        """
        Args:
            config (dict, optional): 모델 설정
        """
        if config is None:
            config = {
                'n_qubits': 4,
                'n_layers': 2,
                'learning_rate': 0.01,
                'epochs': 50,
                'batch_size': 32
            }
        
        self.n_qubits = config.get('n_qubits', 4)
        self.n_layers = config.get('n_layers', 2)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        
        # 양자 회로 생성
        qc = QuantumCircuit(self.n_qubits, self.n_layers)
        self.circuit = qc.create_circuit()
        
        # 파라미터 초기화
        self.weights = pnp.random.randn(
            self.n_layers, self.n_qubits, 2, 
            requires_grad=True
        )
        
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self.is_trained = False
    
    def predict_single(self, x):
        """단일 샘플 예측"""
        output = self.circuit(self.weights, x)
        return 1 if output > 0 else 0
    
    def predict(self, X):
        """다중 샘플 예측"""
        return np.array([self.predict_single(x) for x in X])
    
    def cost_function(self, weights, X, y):
        """
        손실 함수 (평균 제곱 오차)
        
        Args:
            weights: 양자 회로 파라미터
            X, y: 데이터와 레이블
            
        Returns:
            float: 손실 값
        """
        predictions = []
        for x in X:
            output = self.circuit(weights, x)
            predictions.append(output)
        
        predictions = pnp.array(predictions)
        # 레이블을 -1, 1로 변환 (PauliZ 측정 범위)
        y_transformed = 2 * y - 1
        
        return pnp.mean((predictions - y_transformed) ** 2)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        QNN 학습 (경사하강법)
        
        Args:
            X_train, y_train: 학습 데이터
            X_val, y_val: 검증 데이터
        """
        print("\n" + "="*60)
        print("Quantum Neural Network 학습 시작")
        print("="*60)
        print(f"Epochs: {self.epochs}, Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        
        opt = qml.GradientDescentOptimizer(stepsize=self.learning_rate)
        
        n_samples = len(X_train)
        
        for epoch in range(self.epochs):
            # 배치 학습
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_cost = 0
            n_batches = n_samples // self.batch_size
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # 그래디언트 업데이트
                self.weights, cost = opt.step_and_cost(
                    lambda w: self.cost_function(w, X_batch, y_batch),
                    self.weights
                )
                
                epoch_cost += cost
            
            avg_cost = epoch_cost / n_batches
            self.training_history['train_loss'].append(avg_cost)
            
            # 주기적 평가
            if (epoch + 1) % 10 == 0 or epoch == 0:
                train_acc = accuracy_score(y_train, self.predict(X_train))
                self.training_history['train_acc'].append(train_acc)
                
                val_info = ""
                if X_val is not None and y_val is not None:
                    val_acc = accuracy_score(y_val, self.predict(X_val))
                    val_cost = self.cost_function(self.weights, X_val, y_val)
                    self.training_history['val_acc'].append(val_acc)
                    self.training_history['val_loss'].append(val_cost)
                    val_info = f", Val Loss: {val_cost:.4f}, Val Acc: {val_acc*100:.2f}%"
                
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Train Loss: {avg_cost:.4f}, "
                      f"Train Acc: {train_acc*100:.2f}%{val_info}")
        
        self.is_trained = True
        print("\n✓ QNN 학습 완료\n")
    
    def evaluate(self, X, y, verbose=True):
        """
        모델 평가
        
        Args:
            X, y: 테스트 데이터
            verbose (bool): 상세 출력
            
        Returns:
            float: 정확도
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        if verbose:
            print("\n" + "="*60)
            print("Quantum Neural Network Evaluation Results")
            print("="*60)
            print(f"Accuracy: {accuracy*100:.2f}%\n")
            print("Classification Report:")
            print(classification_report(y, y_pred,
                                       target_names=['Not Possible', 'Possible']))
        
        return accuracy
    
    def save_model(self, save_path='./checkpoints/qnn.pkl'):
        """모델 저장"""
        if not self.is_trained:
            raise ValueError("학습된 모델이 없습니다.")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model_data = {
            'weights': self.weights,
            'n_qubits': self.n_qubits,
            'n_layers': self.n_layers,
            'training_history': self.training_history
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ QNN 모델 저장: {save_path}")
    
    def load_model(self, load_path='./checkpoints/qnn.pkl'):
        """저장된 모델 로드"""
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.n_qubits = model_data['n_qubits']
        self.n_layers = model_data['n_layers']
        self.training_history = model_data['training_history']
        self.is_trained = True
        
        # 회로 재생성
        qc = QuantumCircuit(self.n_qubits, self.n_layers)
        self.circuit = qc.create_circuit()
        
        print(f"✓ QNN 모델 로드: {load_path}")
