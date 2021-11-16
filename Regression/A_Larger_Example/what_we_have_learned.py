import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

insurance = pd.read_csv("insurance.csv")
print(insurance)
insurance_one_hot = pd.get_dummies(insurance)
print(insurance_one_hot.head())

X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]
print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X), len(X_train), len(X_test))

# Build a neural network
tf.random.set_seed(42)

# 1.Create model
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# 2.Compile
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=['mae'])
# 3.fit
insurance_model.fit(X_train, y_train, epochs=100)
res = insurance_model.evaluate(X_test, y_test)
print(res)
print(y_train.median())
print(y_train.mean())

insurance_model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

insurance_model_2.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae"])

history = insurance_model_2.fit(X_train, y_train, epochs=100)
res = insurance_model_2.evaluate(X_test, y_test)
print(res)
print(y_train.median())
print(y_train.mean())

# plot history
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()


