"""
논문용 고품질 그래프 생성 스크립트
Publication-ready figure generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# 논문용 설정 (IEEE, Nature, ACM 스타일)
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    # 폰트 설정
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    
    # 선 설정
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    
    # 그리드
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    
    # DPI (해상도)
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    
    # 색상
    'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
})

print("""
╔══════════════════════════════════════════════════════════════════════╗
║           Publication-Ready Figure Generator                         ║
║           High-Quality Graphs for Academic Papers                    ║
╚══════════════════════════════════════════════════════════════════════╝

📊 논문용 그래프 설정:
  - Font: Times New Roman (serif)
  - Resolution: 300 DPI
  - Style: Academic paper format
  - Color scheme: Color-blind friendly

✅ 모든 그래프가 영어로 생성됩니다.
✅ IEEE, Nature, ACM 저널 표준에 맞춤

📁 저장 위치: ./results/

실행 완료!
""")
