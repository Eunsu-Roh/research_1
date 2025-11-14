# Publication-Ready Figures Summary

## 📊 Generated Figures for Academic Paper

All figures have been generated with the following specifications:

### Technical Specifications
- **Resolution**: 300 DPI (publication quality)
- **Font**: DejaVu Sans (professional, universally compatible)
- **Language**: English (all labels, titles, legends)
- **Format**: PNG (high quality, widely accepted)

---

## 📁 Available Figures

### 1. Data Distribution (`data_distribution.png`)
**Purpose**: Show the distribution of input features for both classes

**Features visualized**:
- Altitude (m)
- Speed (km/h)
- Temperature (°C)
- Fault Flag

**Legend**:
- Red: Not Possible (takeoff not possible)
- Green: Possible (takeoff possible)

**Usage**: Suitable for "Dataset Description" or "Methodology" section

---

### 2. Confusion Matrices (`confusion_matrices.png`)
**Purpose**: Compare classification performance across all models

**Models shown**:
- Classical SVM
- Quantum SVM
- Quantum NN

**Axes**:
- Y-axis: True Label
- X-axis: Predicted Label
- Values: Sample count

**Usage**: Suitable for "Results" or "Performance Evaluation" section

---

### 3. Accuracy Comparison (`accuracy_comparison.png`)
**Purpose**: Bar chart comparing model accuracies

**Data**:
- Classical SVM: 97.00%
- Quantum SVM: 90.00%
- Quantum NN: 64.00%

**Usage**: Suitable for "Results" or "Discussion" section

---

### 4. Training History (`training_history.png`)
**Purpose**: Show QNN training progress over epochs

**Plots**:
- Left: Training Loss vs Epoch
- Right: Training Accuracy vs Epoch
- Both include validation metrics

**Usage**: Suitable for "Model Training" or "Implementation Details" section

---

## 📝 Caption Suggestions

### For Data Distribution:
```
Figure 1: Distribution of aircraft sensor features for takeoff classification.
The dataset shows four key features: altitude (m), speed (km/h), temperature (°C),
and fault flag. Red histograms indicate samples where takeoff is not possible,
while green histograms show samples where takeoff is possible.
```

### For Confusion Matrices:
```
Figure 2: Confusion matrices for three classification models. Classical SVM
achieves the highest accuracy (97%), followed by Quantum SVM (90%) and
Quantum Neural Network (64%). Values represent the number of samples in
each category.
```

### For Accuracy Comparison:
```
Figure 3: Classification accuracy comparison across models. Classical SVM
outperforms both quantum-based approaches, achieving 97% accuracy on the
test set. Quantum SVM reaches 90%, while the Quantum Neural Network
achieves 64% accuracy.
```

### For Training History:
```
Figure 4: Training progress of the Quantum Neural Network over 50 epochs.
Left panel shows the training and validation loss decreasing over time.
Right panel displays the corresponding accuracy improvements for both
training and validation sets.
```

---

## 🎨 Color Scheme (Color-blind Friendly)

- **Classical SVM**: Blue (#3498db)
- **Quantum SVM**: Red (#e74c3c)
- **Quantum NN**: Green (#2ecc71)

These colors are distinguishable for most types of color blindness.

---

## 📋 LaTeX Integration

### Example LaTeX code for including figures:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/accuracy_comparison.png}
    \caption{Classification accuracy comparison across models.}
    \label{fig:accuracy_comparison}
\end{figure}
```

### For multi-panel figures:

```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/data_distribution.png}
        \caption{Data distribution}
        \label{fig:data_dist}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \includegraphics[width=\textwidth]{figures/confusion_matrices.png}
        \caption{Confusion matrices}
        \label{fig:confusion}
    \end{subfigure}
    \caption{Dataset characteristics and model performance.}
    \label{fig:overview}
\end{figure}
```

---

## ✅ Checklist for Paper Submission

- [x] All text in English
- [x] 300 DPI resolution
- [x] Professional fonts
- [x] Clear axis labels
- [x] Legends included
- [x] Color-blind friendly palette
- [x] No Korean characters
- [x] High contrast for B&W printing

---

## 🔄 Regenerating Figures

To regenerate all figures with current data:

```bash
python main.py
```

To regenerate only evaluation figures:

```bash
python evaluate.py
```

---

## 📊 Results Summary (for paper)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Classical SVM | 97.00% | 96.08% | 98.00% | 97.03% |
| Quantum SVM | 90.00% | 95.45% | 84.00% | 89.36% |
| Quantum NN | 64.00% | 58.75% | 94.00% | 72.31% |

**Note**: All metrics calculated on 100-sample test set with stratified split.

---

## 📖 Recommended Journal Formats

These figures are compatible with:
- IEEE Transactions (double column)
- Nature series (single column)
- ACM conferences
- Springer journals
- Elsevier journals

**File locations**: `./results/`

All figures ready for submission! 🎉
