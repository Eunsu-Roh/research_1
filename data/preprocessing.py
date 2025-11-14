"""
항공기 데이터 전처리 모듈
PyTorch Dataset 스타일로 구현
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os


class AircraftDataPreprocessor:
    """
    항공기 센서 데이터 전처리 클래스
    - 데이터 생성 또는 로드
    - 정규화
    - Train/Validation/Test 분할
    """
    
    def __init__(self, config):
        """
        Args:
            config (dict): 설정 딕셔너리
        """
        self.config = config
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self):
        """
        합성 항공기 센서 데이터 생성
        실제 NASA 데이터셋이 없을 때 사용
        
        Returns:
            pd.DataFrame: 생성된 데이터프레임
        """
        print("\n" + "="*60)
        print("데이터 생성 중...")
        print("="*60)
        
        n_samples = self.config['data']['n_samples']
        np.random.seed(self.config['data']['random_seed'])
        
        # 특성 생성
        altitude = np.random.normal(
            self.config['data']['altitude_mean'], 
            self.config['data']['altitude_std'], 
            n_samples
        )
        
        speed = np.random.normal(
            self.config['data']['speed_mean'], 
            self.config['data']['speed_std'], 
            n_samples
        )
        
        temperature = np.random.normal(
            self.config['data']['temperature_mean'], 
            self.config['data']['temperature_std'], 
            n_samples
        )
        
        fault_flag = np.random.binomial(
            1, 
            self.config['data']['fault_probability'], 
            n_samples
        )
        
        # 이륙 가능 여부 레이블 생성
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
        
        print(f"✓ 총 샘플 수: {n_samples}")
        print(f"✓ 이륙 가능: {takeoff_possible.sum()} ({takeoff_possible.mean()*100:.1f}%)")
        print(f"✓ 이륙 불가능: {(1-takeoff_possible).sum()} ({(1-takeoff_possible.mean())*100:.1f}%)")
        
        return data
    
    def load_data(self, data_path=None):
        """
        데이터 로드 (CSV 또는 합성 데이터)
        
        Args:
            data_path (str, optional): 데이터 파일 경로
            
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        if data_path and os.path.exists(data_path):
            print(f"데이터 로드 중: {data_path}")
            data = pd.read_csv(data_path)
        else:
            print("데이터 파일이 없습니다. 합성 데이터를 생성합니다.")
            data = self.generate_synthetic_data()
        
        return data
    
    def split_data(self, data):
        """
        데이터를 Train/Validation/Test로 분할
        
        Args:
            data (pd.DataFrame): 전체 데이터
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\n데이터 분할 중...")
        
        features = self.config['data']['features']
        X = data[features].values
        y = data['takeoff_possible'].values
        
        test_size = self.config['data']['test_size']
        val_size = self.config['data']['val_size']
        random_seed = self.config['data']['random_seed']
        
        # Train+Val / Test 분할
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_seed,
            stratify=y
        )
        
        # Train / Validation 분할
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=y_temp
        )
        
        print(f"✓ Train: {len(X_train)} samples")
        print(f"✓ Validation: {len(X_val)} samples")
        print(f"✓ Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_data(self, X_train, X_val, X_test):
        """
        데이터 정규화 (Standardization)
        Train 데이터로 fit, 모든 데이터에 transform
        
        Args:
            X_train, X_val, X_test: 분할된 특성 데이터
            
        Returns:
            tuple: (X_train_scaled, X_val_scaled, X_test_scaled)
        """
        print("\n데이터 정규화 중...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("✓ 정규화 완료 (StandardScaler)")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def save_scaler(self, save_path='./data/scaler.pkl'):
        """
        Scaler 저장 (추론 시 재사용)
        
        Args:
            save_path (str): 저장 경로
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✓ Scaler 저장: {save_path}")
    
    def load_scaler(self, load_path='./data/scaler.pkl'):
        """
        저장된 Scaler 로드
        
        Args:
            load_path (str): 로드 경로
        """
        with open(load_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✓ Scaler 로드: {load_path}")
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test,
                           save_dir='./data/processed'):
        """
        전처리된 데이터 저장
        
        Args:
            save_dir (str): 저장 디렉토리
        """
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
        
        print(f"✓ 전처리 데이터 저장: {save_dir}")
    
    def process_pipeline(self, data_path=None, save_data=True):
        """
        전체 전처리 파이프라인 실행
        
        Args:
            data_path (str, optional): 데이터 파일 경로
            save_data (bool): 전처리 데이터 저장 여부
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test, raw_data)
        """
        # 1. 데이터 로드
        data = self.load_data(data_path)
        
        # 2. 데이터 분할
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(data)
        
        # 3. 정규화
        X_train, X_val, X_test = self.normalize_data(X_train, X_val, X_test)
        
        # 4. 저장
        if save_data:
            self.save_processed_data(X_train, X_val, X_test, 
                                    y_train, y_val, y_test)
            self.save_scaler()
        
        print("\n✓ 전처리 파이프라인 완료\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, data
