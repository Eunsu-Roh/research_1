"""
평가 지표 계산 유틸리티
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    다양한 평가 지표 계산
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        y_proba: 예측 확률 (옵션)
        
    Returns:
        dict: 평가 지표 딕셔너리
    """
    metrics = {}
    
    # 기본 지표
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # 혼동 행렬
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # TN, FP, FN, TP
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negative'] = tn
        metrics['false_positive'] = fp
        metrics['false_negative'] = fn
        metrics['true_positive'] = tp
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC-AUC (확률 예측이 있는 경우)
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = None
    
    return metrics


def print_metrics(metrics, model_name='Model'):
    """
    평가 지표 출력
    
    Args:
        metrics (dict): 평가 지표 딕셔너리
        model_name (str): 모델 이름
    """
    print("\n" + "="*60)
    print(f"{model_name} 평가 지표")
    print("="*60)
    
    print(f"Accuracy:   {metrics['accuracy']*100:6.2f}%")
    print(f"Precision:  {metrics['precision']*100:6.2f}%")
    print(f"Recall:     {metrics['recall']*100:6.2f}%")
    print(f"F1-Score:   {metrics['f1_score']*100:6.2f}%")
    
    if 'specificity' in metrics:
        print(f"Specificity: {metrics['specificity']*100:6.2f}%")
    
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        print(f"ROC-AUC:    {metrics['roc_auc']:6.4f}")
    
    print("\n혼동 행렬:")
    print(metrics['confusion_matrix'])
    
    if 'true_positive' in metrics:
        print(f"\nTP: {metrics['true_positive']}, "
              f"FP: {metrics['false_positive']}, "
              f"TN: {metrics['true_negative']}, "
              f"FN: {metrics['false_negative']}")
    
    print("="*60 + "\n")


def compare_models(results_dict):
    """
    여러 모델의 성능 비교
    
    Args:
        results_dict (dict): {모델명: metrics} 딕셔너리
        
    Returns:
        pd.DataFrame: 비교 표
    """
    import pandas as pd
    
    comparison = {}
    
    for model_name, metrics in results_dict.items():
        comparison[model_name] = {
            'Accuracy': f"{metrics['accuracy']*100:.2f}%",
            'Precision': f"{metrics['precision']*100:.2f}%",
            'Recall': f"{metrics['recall']*100:.2f}%",
            'F1-Score': f"{metrics['f1_score']*100:.2f}%"
        }
    
    df = pd.DataFrame(comparison).T
    
    print("\n" + "="*60)
    print("모델 성능 비교")
    print("="*60)
    print(df)
    print("="*60 + "\n")
    
    return df
