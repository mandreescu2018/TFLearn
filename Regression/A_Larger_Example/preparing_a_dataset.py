import tensorflow as tf
import pandas as pd

insurance = pd.read_csv("insurance.csv")
print(insurance)
insurance_one_hot = pd.get_dummies(insurance)

