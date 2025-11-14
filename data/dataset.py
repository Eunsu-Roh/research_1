"""
PyTorch 스타일 Dataset 클래스
양자 머신러닝에서도 동일한 패턴 사용
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class AircraftDataset(Dataset):
    """
    PyTorch Dataset 인터페이스를 따르는 항공기 데이터셋
    양자 모델에서도 사용 가능
    """
    
    def __init__(self, X, y, transform=None):
        """
        Args:
            X (np.ndarray): 특성 데이터
            y (np.ndarray): 레이블 데이터
            transform (callable, optional): 변환 함수
        """
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        인덱스에 해당하는 샘플 반환
        
        Args:
            idx (int): 샘플 인덱스
            
        Returns:
            tuple: (features, label)
        """
        features = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return features, label
    
    def get_features(self):
        """전체 특성 데이터 반환"""
        return self.X
    
    def get_labels(self):
        """전체 레이블 데이터 반환"""
        return self.y
    
    def get_batch(self, indices):
        """
        배치 데이터 반환
        
        Args:
            indices (list or np.ndarray): 배치 인덱스
            
        Returns:
            tuple: (batch_features, batch_labels)
        """
        return self.X[indices], self.y[indices]


class QuantumDataLoader:
    """
    양자 모델을 위한 커스텀 데이터 로더
    PyTorch DataLoader와 유사한 인터페이스
    """
    
    def __init__(self, dataset, batch_size=32, shuffle=True):
        """
        Args:
            dataset (AircraftDataset): 데이터셋
            batch_size (int): 배치 크기
            shuffle (bool): 셔플 여부
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        """배치 이터레이터"""
        n_samples = len(self.dataset)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield self.dataset.get_batch(batch_indices)
    
    def __len__(self):
        """배치 개수 반환"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
