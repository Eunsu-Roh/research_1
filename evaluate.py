"""
모델 평가 스크립트
"""

import yaml
import numpy as np
from models.classical_svm import ClassicalSVM
from models.qsvm import QuantumSVM
from models.qnn import QuantumNeuralNetwork
from utils.visualization import ResultVisualizer
from utils.metrics import calculate_metrics, print_metrics, compare_models
import os


def load_config(config_path='./configs/config.yaml'):
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_test_data(data_dir='./data/processed'):
    """전처리된 테스트 데이터 로드"""
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    return X_test, y_test


def evaluate_all_models(config, X_test, y_test):
    """
    모든 모델 평가
    
    Returns:
        dict: {모델명: (정확도, 예측값, 지표)} 딕셔너리
    """
    results = {}
    
    # 1. 고전 SVM 평가
    print("\n" + "="*70)
    print("1. 고전 SVM 평가")
    print("="*70)
    try:
        classical_svm = ClassicalSVM(config['models']['classical_svm'])
        classical_svm.load_model()
        
        y_pred = classical_svm.predict(X_test)
        acc = classical_svm.evaluate(X_test, y_test, verbose=False)
        
        metrics = calculate_metrics(y_test, y_pred)
        print_metrics(metrics, 'Classical SVM')
        
        results['Classical SVM'] = (acc, y_pred, metrics)
    except Exception as e:
        print(f"⚠ 고전 SVM 로드 실패: {e}")
    
    # 2. Quantum SVM 평가
    print("\n" + "="*70)
    print("2. Quantum SVM 평가")
    print("="*70)
    try:
        qsvm = QuantumSVM(config['models']['qsvm'])
        qsvm.load_model()
        
        y_pred = qsvm.predict(X_test)
        acc = qsvm.evaluate(X_test, y_test, verbose=False)
        
        metrics = calculate_metrics(y_test, y_pred)
        print_metrics(metrics, 'Quantum SVM')
        
        results['Quantum SVM'] = (acc, y_pred, metrics)
    except Exception as e:
        print(f"⚠ Quantum SVM 로드 실패: {e}")
    
    # 3. Quantum Neural Network 평가
    print("\n" + "="*70)
    print("3. Quantum Neural Network 평가")
    print("="*70)
    try:
        qnn = QuantumNeuralNetwork(config['models']['qnn'])
        qnn.load_model()
        
        y_pred = qnn.predict(X_test)
        acc = qnn.evaluate(X_test, y_test, verbose=False)
        
        metrics = calculate_metrics(y_test, y_pred)
        print_metrics(metrics, 'Quantum NN')
        
        results['Quantum NN'] = (acc, y_pred, metrics)
    except Exception as e:
        print(f"⚠ Quantum NN 로드 실패: {e}")
    
    return results


def visualize_results(config, y_test, results):
    """결과 시각화 및 저장"""
    visualizer = ResultVisualizer(config['output']['results_dir'])
    
    # 정확도 딕셔너리 생성
    accuracies = {name: acc for name, (acc, _, _) in results.items()}
    
    # 예측값 딕셔너리 생성
    predictions = {name: pred for name, (_, pred, _) in results.items()}
    
    # 지표 딕셔너리 생성
    metrics_dict = {name: metrics for name, (_, _, metrics) in results.items()}
    
    # 시각화
    if config['output']['save_plots']:
        visualizer.plot_accuracy_comparison(accuracies)
        visualizer.plot_confusion_matrices(y_test, predictions)
    
    # 요약 리포트 저장
    visualizer.save_summary_report(accuracies)
    
    # 모델 비교 출력
    compare_models(metrics_dict)


def main():
    """메인 평가 파이프라인"""
    print("\n" + "="*70)
    print("항공기 이륙 가능 여부 분류 - 모델 평가")
    print("="*70)
    
    # 설정 로드
    config = load_config()
    print("\n✓ 설정 로드 완료")
    
    # 테스트 데이터 로드
    print("\n테스트 데이터 로드 중...")
    X_test, y_test = load_test_data()
    print(f"✓ 테스트 샘플 수: {len(X_test)}")
    
    # 모든 모델 평가
    results = evaluate_all_models(config, X_test, y_test)
    
    # 결과 시각화
    if results:
        visualize_results(config, y_test, results)
    
    print("\n" + "="*70)
    print("모든 모델 평가 완료!")
    print("="*70)
    print(f"\n결과 저장 위치: {config['output']['results_dir']}")
    
    return results


if __name__ == "__main__":
    main()
