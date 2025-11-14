"""
고전 Support Vector Machine 모델
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


class ClassicalSVM:
    """
    고전적인 SVM 분류기
    sklearn 기반 구현
    """
    
    def __init__(self, config=None):
        """
        Args:
            config (dict, optional): 모델 설정
        """
        if config is None:
            config = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}
        
        self.config = config
        self.model = SVC(
            kernel=config.get('kernel', 'rbf'),
            C=config.get('C', 1.0),
            gamma=config.get('gamma', 'scale'),
            random_state=42
        )
        self.is_trained = False
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        모델 학습
        
        Args:
            X_train: 학습 데이터
            y_train: 학습 레이블
            X_val: 검증 데이터 (옵션)
            y_val: 검증 레이블 (옵션)
        """
        print("\n" + "="*60)
        print(f"고전 SVM 학습 시작 (커널: {self.config['kernel']})")
        print("="*60)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # 학습 데이터 정확도
        train_acc = self.evaluate(X_train, y_train, verbose=False)
        print(f"✓ 학습 데이터 정확도: {train_acc*100:.2f}%")
        
        # 검증 데이터 정확도
        if X_val is not None and y_val is not None:
            val_acc = self.evaluate(X_val, y_val, verbose=False)
            print(f"✓ 검증 데이터 정확도: {val_acc*100:.2f}%")
        
        print("✓ 학습 완료\n")
        
    def predict(self, X):
        """
        예측 수행
        
        Args:
            X: 입력 데이터
            
        Returns:
            np.ndarray: 예측 레이블
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. train() 먼저 호출하세요.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        확률 예측 (SVM은 decision_function 사용)
        
        Args:
            X: 입력 데이터
            
        Returns:
            np.ndarray: 결정 함수 값
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        return self.model.decision_function(X)
    
    def evaluate(self, X, y, verbose=True):
        """
        모델 평가
        
        Args:
            X: 테스트 데이터
            y: 실제 레이블
            verbose (bool): 상세 출력 여부
            
        Returns:
            float: 정확도
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        if verbose:
            print("\n" + "="*60)
            print("Classical SVM Evaluation Results")
            print("="*60)
            print(f"Accuracy: {accuracy*100:.2f}%\n")
            print("Classification Report:")
            print(classification_report(y, y_pred, 
                                       target_names=['Not Possible', 'Possible']))
        
        return accuracy
    
    def save_model(self, save_path='./checkpoints/classical_svm.pkl'):
        """
        모델 저장
        
        Args:
            save_path (str): 저장 경로
        """
        if not self.is_trained:
            raise ValueError("학습된 모델이 없습니다.")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ 모델 저장: {save_path}")
    
    def load_model(self, load_path='./checkpoints/classical_svm.pkl'):
        """
        저장된 모델 로드
        
        Args:
            load_path (str): 로드 경로
        """
        with open(load_path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"✓ 모델 로드: {load_path}")
