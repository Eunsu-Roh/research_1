# Quantum Aircraft Classifier

A takeoff feasibility classification system using aircraft sensor data  
**A comparative study of classical machine learning vs quantum machine learning**

## 📁 Project Structure

```
research_presentation/
├── configs/              # Configuration files
│   └── config.yaml       # Hyperparameters and experiment settings
│
├── data/                 # Data-related modules
│   ├── __init__.py
│   ├── preprocessing.py  # Data preprocessing (load, normalize, split)
│   ├── dataset.py        # PyTorch-style Dataset class
│   └── processed/        # Preprocessed data storage (auto-generated)
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       └── y_test.npy
│
├── models/               # Model definitions
│   ├── __init__.py
│   ├── classical_svm.py  # Classical Support Vector Machine
│   ├── qsvm.py           # Quantum SVM (quantum kernel)
│   └── qnn.py            # Quantum Neural Network
│
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── visualization.py  # Visualization (plots, confusion matrix)
│   └── metrics.py        # Evaluation metric calculation
│
├── checkpoints/          # Trained model storage (auto-generated)
│   ├── classical_svm.pkl
│   ├── qsvm.pkl
│   └── qnn.pkl
│
├── results/              # Experiment results (auto-generated)
│   ├── data_distribution.png
│   ├── confusion_matrices.png
│   ├── accuracy_comparison.png
│   ├── training_history.png
│   └── summary_report.txt
│
├── main.py               # Run full pipeline
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── lab.py                # Integrated version (legacy file)
└── README.md             # Project documentation (this file)
```

## 🚀 Quick Start

### 1. Environment Setup

```powershell
conda activate env
pip install numpy pandas matplotlib seaborn scikit-learn pennylane pyyaml
```

### 2. Run the Full Pipeline

```powershell
python main.py
```

This command automatically performs the following:
- Data generation and preprocessing
- Train/Validation/Test split
- Train 3 models (Classical SVM, QSVM, QNN)
- Model evaluation and comparison
- Result visualization and saving

### 3. Run Individually

#### Run training only
```powershell
python train.py
```

#### Run evaluation only (requires trained models)
```powershell
python evaluate.py
```

## 📊 Dataset

### Features
- **Altitude**: Aircraft altitude (m)
- **Speed**: Aircraft speed (km/h)
- **Temperature**: Outside air temperature (°C)
- **Fault Flag**: System fault status (0/1)

### Label
- **Takeoff Feasible (1)**: Safe takeoff condition
- **Takeoff Not Feasible (0)**: Unsafe takeoff condition

### Data Split
- **Train**: 70% (training)
- **Validation**: 10% (validation)
- **Test**: 20% (final evaluation)

## 🤖 Models

### 1. Classical SVM (Classical SVM)
- **Kernel**: RBF (Radial Basis Function)
- **Purpose**: Provide baseline performance
- **Implementation**: Scikit-learn

### 2. Quantum SVM (QSVM)
- **Quantum Kernel**: Based on Angle Encoding
- **Number of Qubits**: 4
- **Purpose**: Validate representation power of quantum kernels
- **Implementation**: PennyLane + Scikit-learn

### 3. Quantum Neural Network (QNN)
- **Architecture**: Variational Quantum Classifier
- **Encoding**: Angle Encoding
- **Depth**: 2 layers
- **Optimization**: Gradient Descent
- **Implementation**: PennyLane

## ⚙️ Configuration (config.yaml)

Main configuration parameters:

```yaml
data:
  n_samples: 500        # Number of samples to generate
  test_size: 0.2        # Test set ratio
  val_size: 0.1         # Validation set ratio

models:
  qnn:
    epochs: 50          # Training epochs
    batch_size: 32      # Batch size
    learning_rate: 0.01 # Learning rate
```

## 📈 Output Results

### 1. Console Output
- Data generation information
- Training progress
- Accuracy and classification report by model
- Best-performing model

### 2. Saved Files

#### Plots (`results/`)
- `data_distribution.png`: Data distribution by feature
- `confusion_matrices.png`: Confusion matrices by model
- `accuracy_comparison.png`: Accuracy comparison
- `training_history.png`: QNN training history

#### Model Checkpoints (`checkpoints/`)
- `classical_svm.pkl`: Trained classical SVM
- `qsvm.pkl`: Trained QSVM
- `qnn.pkl`: Trained QNN

#### Data (`data/processed/`)
- Preprocessed Train/Val/Test data
- Scaler object (reused during inference)

## 🔬 Key Features

### PyTorch-style structure
- **Modularity**: Separation of data, models, and utilities
- **Dataset class**: Implements the PyTorch Dataset interface
- **Checkpoints**: Save/load trained models
- **Reproducibility**: Fixed seed and config file management

### Quantum machine learning implementation
- **Angle Encoding**: Converts classical data to quantum states
- **Variational Circuit**: Trainable parameterized layers
- **Quantum Kernel**: SVM based on quantum similarity
- **Hybrid**: Classical-quantum combined architecture

## 📝 Usage Example

### Run with custom config
```python
# After creating custom_config.yaml
python main.py --config ./configs/custom_config.yaml
```

### Train a single model
```python
from models.qnn import QuantumNeuralNetwork
from data.preprocessing import AircraftDataPreprocessor

# Prepare data
config = load_config()
preprocessor = AircraftDataPreprocessor(config)
X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocessor.process_pipeline()

# Train QNN
qnn = QuantumNeuralNetwork(config['models']['qnn'])
qnn.train(X_train, y_train, X_val, y_val)
qnn.save_model()

# Evaluate
accuracy = qnn.evaluate(X_test, y_test)
```

## 🎯 Extension Ideas

1. **Use real-world data**: Integrate NASA Turbofan Dataset
2. **Multi-class classification**: Normal/Warning/Risk 3-level classification
3. **Ensemble**: Quantum-classical hybrid ensemble
4. **Optimization**: Hyperparameter tuning
5. **Reinforcement Learning**: Route decision support system

## 📚 References

- **PennyLane**: https://pennylane.ai/
- **Qiskit**: https://qiskit.org/
- **NASA Dataset**: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- **Quantum Machine Learning**: https://arxiv.org/abs/1611.09347

## 📄 License

This project was created for research and educational purposes.

---

**Author**: Undergraduate Researcher  
**Date**: November 14, 2025
