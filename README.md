HEAD
# heart_disease_bayes
=======
# Bayes to the Future: Predicting Heart Disease

This project uses Bayesian Networks to model and predict heart disease risk.

## 🧪 Dataset
Simulated patient data: `heart_disease.csv`

## 🧹 Steps
- Removed duplicates and missing values
- Min-max normalization on numeric columns
- Discretized features for Bayesian modeling
- Built a Bayesian Network using `pgmpy`
- Queried probabilities of heart disease risk

## 📊 Example Inference
- P(target | age = 'old')
- P(chol | target = 1)

## 🔧 Run Instructions
```bash
pip install pandas scikit-learn pgmpy matplotlib networkx
python heart_disease_bayes.py
65722f2 (Initial commit - heart disease bayesian model)
