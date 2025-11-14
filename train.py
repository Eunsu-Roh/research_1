"""
모델 학습 스크립트
"""

import yaml
import numpy as np
from data.preprocessing import AircraftDataPreprocessor
from data.dataset import AircraftDataset
from models.classical_svm import ClassicalSVM
from models.qsvm import QuantumSVM
from models.qnn import QuantumNeuralNetwork
from utils.visualization import ResultVisualizer
from utils.metrics import calculate_metrics, print_metrics
import os


def load_config(config_path='./configs/config.yaml'):
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_all_models(config, X_train, X_val, y_train, y_val):
    """
    모든 모델 학습
    
    Returns:
        dict: {모델명: 모델 객체} 딕셔너리
    """
    models = {}
    
    # 1. 고전 SVM
    print("\n" + "="*70)
    print("1. 고전 SVM 학습")
    print("="*70)
    classical_svm = ClassicalSVM(config['models']['classical_svm'])
    classical_svm.train(X_train, y_train, X_val, y_val)
    classical_svm.save_model()
    models['Classical SVM'] = classical_svm
    
    # 2. Quantum SVM
    print("\n" + "="*70)
    print("2. Quantum SVM 학습")
    print("="*70)
    qsvm = QuantumSVM(config['models']['qsvm'])
    qsvm.train(X_train, y_train, X_val, y_val)
    qsvm.save_model()
    models['Quantum SVM'] = qsvm
    
    # 3. Quantum Neural Network
    print("\n" + "="*70)
    print("3. Quantum Neural Network 학습")
    print("="*70)
    qnn = QuantumNeuralNetwork(config['models']['qnn'])
    qnn.train(X_train, y_train, X_val, y_val)
    qnn.save_model()
    models['Quantum NN'] = qnn
    
    # QNN 학습 과정 시각화
    if config['output']['save_plots']:
        visualizer = ResultVisualizer(config['output']['results_dir'])
        visualizer.plot_training_history(qnn.training_history, 'QNN')
    
    return models


def main():
    """메인 학습 파이프라인"""
    print("\n" + "="*70)
    print("항공기 이륙 가능 여부 분류 - 모델 학습")
    print("="*70)
    
    # 설정 로드
    config = load_config()
    print("\n✓ 설정 로드 완료")
    
    # 데이터 전처리
    preprocessor = AircraftDataPreprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test, raw_data = \
        preprocessor.process_pipeline(save_data=True)
    
    # 데이터 분포 시각화
    if config['output']['save_plots']:
        visualizer = ResultVisualizer(config['output']['results_dir'])
        visualizer.plot_data_distribution(raw_data)
    
    # 모든 모델 학습
    models = train_all_models(config, X_train, X_val, y_train, y_val)
    
    print("\n" + "="*70)
    print("모든 모델 학습 완료!")
    print("="*70)
    print(f"\n학습된 모델 저장 위치: {config['training']['checkpoint_dir']}")
    print(f"결과 저장 위치: {config['output']['results_dir']}")
    
    return models


if __name__ == "__main__":
    main()
