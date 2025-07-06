import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


X = np.array([
    [50, 2],
    [60, 3],
    [70, 3],
    [80, 4],
    [90, 4],
    [100, 5],
    [110, 5],
    [120, 6],
    [130, 6],
    [140, 7]
])


y = np.array([150, 180, 200, 220, 240, 260, 280, 300, 320, 340])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Tahmin edilen fiyatlar:", y_pred)
print("Gerçek fiyatlar:", y_test)
print(f"Ortalama kare hata (MSE): {mse:.2f}")
print(f"R^2 skoru: {r2:.2f}")
