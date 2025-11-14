"""
메인 실행 파일
학습과 평가를 통합 실행
"""

import yaml
import argparse
from train import train_all_models
from evaluate import evaluate_all_models, visualize_results
from data.preprocessing import AircraftDataPreprocessor
from utils.visualization import ResultVisualizer


def load_config(config_path='./configs/config.yaml'):
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_full_pipeline(config):
    """전체 파이프라인 실행 (데이터 전처리 → 학습 → 평가)"""
    
    print("\n" + "="*70)
    print("항공기 이륙 가능 여부 분류 시스템")
    print("고전 SVM vs 양자 분류기 (QSVM, QNN)")
    print("="*70)
    
    # 1. 데이터 전처리
    print("\n" + "="*70)
    print("STEP 1: 데이터 전처리")
    print("="*70)
    
    preprocessor = AircraftDataPreprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test, raw_data = \
        preprocessor.process_pipeline(save_data=True)
    
    # 데이터 분포 시각화
    if config['output']['save_plots']:
        visualizer = ResultVisualizer(config['output']['results_dir'])
        visualizer.plot_data_distribution(raw_data)
    
    # 2. 모델 학습
    print("\n" + "="*70)
    print("STEP 2: 모델 학습")
    print("="*70)
    
    models = train_all_models(config, X_train, X_val, y_train, y_val)
    
    # 3. 모델 평가
    print("\n" + "="*70)
    print("STEP 3: 모델 평가")
    print("="*70)
    
    results = evaluate_all_models(config, X_test, y_test)
    
    # 4. 결과 시각화
    if results:
        visualize_results(config, y_test, results)
    
    # 최종 요약
    print("\n" + "="*70)
    print("전체 파이프라인 완료!")
    print("="*70)
    
    accuracies = {name: acc for name, (acc, _, _) in results.items()}
    
    print("\n[최종 결과 요약]")
    print("-" * 70)
    for model_name, acc in accuracies.items():
        print(f"  {model_name:20s}: {acc*100:6.2f}%")
    
    best_model = max(accuracies, key=accuracies.get)
    best_acc = accuracies[best_model]
    print("-" * 70)
    print(f"\n🏆 최고 성능 모델: {best_model} ({best_acc*100:.2f}%)")
    
    print(f"\n📁 결과 저장 위치:")
    print(f"   - 체크포인트: {config['training']['checkpoint_dir']}")
    print(f"   - 결과 그래프: {config['output']['results_dir']}")
    print(f"   - 전처리 데이터: ./data/processed")
    
    print("\n" + "="*70 + "\n")
    
    return models, results


def main():
    """
    메인 함수
    
    사용 예시:
        python main.py                    # 전체 파이프라인 실행
        python main.py --config custom.yaml  # 커스텀 설정 파일 사용
    """
    parser = argparse.ArgumentParser(
        description='항공기 이륙 가능 여부 분류 - 양자 머신러닝'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/config.yaml',
        help='설정 파일 경로'
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    
    # 전체 파이프라인 실행
    models, results = run_full_pipeline(config)
    
    return models, results


if __name__ == "__main__":
    main()
