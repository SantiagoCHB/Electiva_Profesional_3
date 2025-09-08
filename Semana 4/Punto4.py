from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

X = pd.DataFrame({"x1":[6,8,6,4,5,7,8],
                  "x2":[3,1,2,2,1,4,5],
                  "x3":[22,35,41,3,18,18,9]})
y = pd.Series([9,1,5,3,9,9,6])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

modelo_local = LinearRegression().fit(X_train, y_train)

def predecir(modelo, datos):
    X_train, X_test, y_train, y_test = datos
    return modelo.predict(X_test)

print(predecir(modelo_local, (X_train, X_test, y_train, y_test)))