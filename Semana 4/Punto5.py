from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

X = pd.DataFrame({"x1":[7,6,5,3,5],
                  "x2":[5,4,4,4,4],
                  "x3":[47,50,11,42,1]})
y = pd.Series([10,3,7,6,8])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

modelo_local = LinearRegression().fit(X_train, y_train)

def calcular_precision(modelo, X_test, y_test):
    return np.float64(modelo.score(X_test, y_test))

print(calcular_precision(modelo_local, X_test, y_test))