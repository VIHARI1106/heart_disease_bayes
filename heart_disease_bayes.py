import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Load and Clean Dataset
df = pd.read_csv("heart_disease.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Step 2: Normalize numeric columns
numeric_cols = df.select_dtypes(include='number').columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Step 3: Discretize (for pgmpy which needs discrete data)
df['age'] = pd.cut(df['age'], bins=3, labels=["young", "mid", "old"])
df['chol'] = pd.cut(df['chol'], bins=3, labels=["low", "mid", "high"])
df['thalach'] = pd.cut(df['thalach'], bins=3, labels=["low", "mid", "high"])

# Step 4: Define Bayesian Network Structure
model = BayesianNetwork([
    ('age', 'fbs'),
    ('fbs', 'target'),
    ('target', 'chol'),
    ('target', 'thalach')
])

# Step 5: Train Model
model.fit(df, estimator=MaximumLikelihoodEstimator)

# Step 6: Visualize the Bayesian Network
plt.figure(figsize=(8, 6))
nx.draw(model, with_labels=True, node_color='skyblue', node_size=2000, font_size=12)
plt.title("Bayesian Network Structure")
plt.show()

# Step 7: Inference
infer = VariableElimination(model)

# Example Query 1: P(heart disease | age = 'old')
print("\nP(target | age = 'old'):")
print(infer.query(['target'], evidence={'age': 'old'}))

# Example Query 2: P(chol | target = 1)
print("\nP(chol | target = 1):")
print(infer.query(['chol'], evidence={'target': 1}))
