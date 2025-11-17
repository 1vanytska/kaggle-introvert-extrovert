# Personality Classification with Deep Neural Network (DNN)

## 📌 Project Overview
This project demonstrates the use of a Deep Neural Network (DNN) to classify personality traits into **Introvert vs. Extrovert** based on behavioral and categorical features.  
The workflow emphasizes **reproducibility, fairness, and reliability** through a robust preprocessing pipeline and careful hyperparameter tuning.

---

## 📦 Requirements
- Python 3.9+
- TensorFlow
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

On Kaggle, most of these libraries are already pre-installed.  
If needed, install missing dependencies:
```bash
!pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

🚀 Quick Start (Kaggle)
- Open the notebook in Kaggle:
notebooks/introvertsextroverts.ipynb
- Run all cells in order (Cell → Run All).
- The notebook will:
- Load and preprocess the dataset
- Train the DNN model
- Evaluate performance (Accuracy, AUC)
- Output results and class weights
You can also execute the notebook programmatically:
```
!jupyter nbconvert --to notebook --execute notebooks/introvertsextroverts.ipynb --output results.ipynb
```

## ⚙️ Preprocessing Pipelin
- IDs and target separated (id, Personality)
- Feature types identified (numeric vs. categorical)
- Missing values handled
- Numeric: Iterative Imputer
- Categorical: Most frequent value
- Encoding
- Categorical: Ordinal Encoding
- Target: Label Encoding (Introvert = 0, Extrovert = 1)
- Scaling: StandardScaler applied to all features
- Class imbalance: Class weights computed for balanced training

---

## 🧠 Model Architecture
- Input layer (feature dimension)
- Dropout (0.1) for regularization
- Dense layers: 64 → 32 → 16 → 8 (ReLU activation, Batch Normalization)
- Output layer: 1 neuron (Sigmoid activation)

---

## 🔧 Training Setup
- Train/Validation split: 80/20 (stratified)
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy, AUC
- Epochs: 30
- Batch size: 32
- Class weights applied
Callbacks:
- EarlyStopping (patience = 10, restore best weights)
- ReduceLROnPlateau (factor = 0.5, patience = 5)

---

## 📊 Results
- Reliable preprocessing pipeline ensured reproducibility
- DNN achieved strong performance on personality classification
- Class weights improved fairness across classes

---

## 📝 Conclusion
Deep learning methods can effectively capture complex patterns in personality-related data.
This project highlights the importance of data preprocessing, fairness handling, and reproducibility in building reliable ML pipelines.
