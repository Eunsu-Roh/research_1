"""
시각화 유틸리티
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Academic paper font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 13


class ResultVisualizer:
    """
    실험 결과 시각화 클래스
    """
    
    def __init__(self, save_dir='./results'):
        """
        Args:
            save_dir (str): 결과 저장 디렉토리
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_data_distribution(self, data, save_name='data_distribution.png'):
        """
        데이터 분포 시각화
        
        Args:
            data (pd.DataFrame): 원본 데이터
            save_name (str): 저장 파일명
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        features = ['altitude', 'speed', 'temperature', 'fault_flag']
        feature_names = ['Altitude (m)', 'Speed (km/h)', 'Temperature (°C)', 'Fault Flag']
        
        for idx, (feature, name) in enumerate(zip(features, feature_names)):
            ax = axes[idx // 2, idx % 2]
            
            data_0 = data[data['takeoff_possible'] == 0][feature]
            data_1 = data[data['takeoff_possible'] == 1][feature]
            
            ax.hist(data_0, alpha=0.5, label='Not Possible', bins=20, color='red')
            ax.hist(data_1, alpha=0.5, label='Possible', bins=20, color='green')
            
            ax.set_xlabel(name, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{name} Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 데이터 분포 그래프 저장: {save_path}")
        plt.close()
    
    def plot_confusion_matrices(self, y_test, predictions_dict, 
                                save_name='confusion_matrices.png'):
        """
        여러 모델의 혼동 행렬 시각화
        
        Args:
            y_test: 실제 레이블
            predictions_dict (dict): {모델명: 예측값} 딕셔너리
            save_name (str): 저장 파일명
        """
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=11)
            axes[idx].set_xlabel('Predicted Label', fontsize=11)
            axes[idx].set_xticklabels(['Not Possible', 'Possible'])
            axes[idx].set_yticklabels(['Not Possible', 'Possible'])
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 혼동 행렬 저장: {save_path}")
        plt.close()
    
    def plot_accuracy_comparison(self, accuracies_dict, 
                                 save_name='accuracy_comparison.png'):
        """
        모델별 정확도 비교 막대 그래프
        
        Args:
            accuracies_dict (dict): {모델명: 정확도} 딕셔너리
            save_name (str): 저장 파일명
        """
        models = list(accuracies_dict.keys())
        accuracies = list(accuracies_dict.values())
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=colors[:len(models)], 
                      edgecolor='black', linewidth=1.5)
        
        # 막대 위에 정확도 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height*100:.2f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.ylim(0, 1.1)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 정확도 비교 그래프 저장: {save_path}")
        plt.close()
    
    def plot_training_history(self, history, model_name='QNN',
                             save_name='training_history.png'):
        """
        학습 과정 시각화 (Loss, Accuracy)
        
        Args:
            history (dict): 학습 히스토리
            model_name (str): 모델 이름
            save_name (str): 저장 파일명
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss 그래프
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss', 
                        marker='o', linewidth=2)
            if 'val_loss' in history and len(history['val_loss']) > 0:
                # Validation loss는 일부 epoch에만 기록되므로 x축 조정
                val_epochs = np.linspace(0, len(history['train_loss'])-1, 
                                        len(history['val_loss']))
                axes[0].plot(val_epochs, history['val_loss'], 
                           label='Validation Loss', marker='s', linewidth=2)
            
            axes[0].set_xlabel('Epoch', fontsize=11)
            axes[0].set_ylabel('Loss', fontsize=11)
            axes[0].set_title(f'{model_name} Training Loss', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
        
        # Accuracy 그래프
        if 'train_acc' in history:
            train_epochs = np.linspace(0, len(history['train_loss'])-1, 
                                      len(history['train_acc']))
            axes[1].plot(train_epochs, history['train_acc'], 
                        label='Train Accuracy', marker='o', linewidth=2)
            
            if 'val_acc' in history and len(history['val_acc']) > 0:
                val_epochs = np.linspace(0, len(history['train_loss'])-1, 
                                        len(history['val_acc']))
                axes[1].plot(val_epochs, history['val_acc'], 
                           label='Validation Accuracy', marker='s', linewidth=2)
            
            axes[1].set_xlabel('Epoch', fontsize=11)
            axes[1].set_ylabel('Accuracy', fontsize=11)
            axes[1].set_title(f'{model_name} Training Accuracy', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 학습 과정 그래프 저장: {save_path}")
        plt.close()
    
    def save_summary_report(self, accuracies_dict, save_name='summary_report.txt'):
        """
        실험 결과 요약 리포트 저장
        
        Args:
            accuracies_dict (dict): {모델명: 정확도} 딕셔너리
            save_name (str): 저장 파일명
        """
        save_path = os.path.join(self.save_dir, save_name)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("항공기 이륙 가능 여부 분류 실험 결과 요약\n")
            f.write("="*70 + "\n\n")
            
            f.write("모델별 정확도:\n")
            f.write("-"*70 + "\n")
            for model, acc in accuracies_dict.items():
                f.write(f"{model:20s}: {acc*100:6.2f}%\n")
            
            f.write("\n")
            
            # 최고 성능 모델
            best_model = max(accuracies_dict, key=accuracies_dict.get)
            best_acc = accuracies_dict[best_model]
            
            f.write(f"최고 성능 모델: {best_model} ({best_acc*100:.2f}%)\n")
            f.write("="*70 + "\n")
        
        print(f"✓ 요약 리포트 저장: {save_path}")
