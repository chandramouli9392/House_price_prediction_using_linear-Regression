import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

data = {
    'Area': [1000, 1500, 1800, 2400, 3000],
    'Bedrooms': [2, 3, 3, 4, 5],
    'Price': [150000, 200000, 250000, 310000, 400000]
}
df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Decision Tree': DecisionTreeRegressor(),
    'SVR': SVR()
}

def run_model():
    model_name = model_choice.get()
    model = models[model_name]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    result_label.config(text=f"Model: {model_name}\nMSE: {round(mse, 2)}")

    plt.scatter(X['Area'], y, color='blue')
    plt.scatter(X_test['Area'], predictions, color='green')
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title(f"{model_name} Prediction")
    plt.show()

app = tk.Tk()
app.title("House Price Predictor")
app.geometry("400x250")

tk.Label(app, text="Select Model", font=("Arial", 12)).pack(pady=10)

model_choice = ttk.Combobox(app, values=list(models.keys()), font=("Arial", 12))
model_choice.pack()
model_choice.current(0)

tk.Button(app, text="Run Prediction", command=run_model, font=("Arial", 12), bg="skyblue").pack(pady=20)

result_label = tk.Label(app, text="", font=("Arial", 12))
result_label.pack()

app.mainloop()
