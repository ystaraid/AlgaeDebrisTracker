import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import ttk

# Data definitions and descriptions
# Regional Coordinates: Data about monitoring locations with latitude and longitude
regional_coordinates = pd.DataFrame({
    'ID': [18, 1, 36, 26, 9],
    'Monitoring Location': ['Gangneung Songjeong', 'Ganghwa Yeochari', 'Geoje Dumo', 'Gochang Dongho', 'Goheung Sinheung'],
    'Latitude': [37.780347, 37.609196, 34.980383, 35.510152, 34.584309],
    'Longitude': [128.937288, 126.381198, 128.694822, 126.478666, 127.145063]
})

# Phase-Based Waste: Data about waste collected during different phases of monitoring
phase_based_waste = pd.DataFrame({
    'Year': [2008, 2009, 2010, 2011, 2012],
    'Phase 1 Count (EA)': [None, 9463.0, 7903.0, 6518.0, 5802.0],
    'Phase 1 Weight (kg)': [None, 1998.0, 884.0, 1301.2, 1156.4],
    'Phase 2 Count (EA)': [10111, 11028, 7329, 7476, 7952],
    'Phase 2 Weight (kg)': [1636.1, 1845.9, 1021.2, 1054.3, 2789.4]
})

# Yearly Waste: Yearly aggregated waste data
yearly_waste = pd.DataFrame({
    'Year': [2008, 2009, 2010, 2011, 2012],
    'Count (EA)': [50588, 54902, 64406, 52323, 52678],
    'Weight (kg)': [9362.5, 10161.6, 9888.6, 11422.1, 9823.6]
})

# Algae Data: Simulated data for algae density
algae_data = pd.DataFrame({
    'Year': [2008, 2009, 2010, 2011, 2012],
    'Region': ['East Sea', 'South Sea', 'West Sea', 'East Sea', 'South Sea'],
    'Density (g/m2)': [120, 135, 140, 125, 130]
})

# Function to perform exploratory data analysis (EDA)
def perform_eda():
    grouped = yearly_waste.groupby("Year")[["Count (EA)", "Weight (kg)"]].sum()
    grouped.plot(subplots=True, layout=(2, 1), figsize=(10, 8), marker="o", title="Yearly Waste Data Analysis")
    plt.show()
    return grouped

# Function to analyze regional waste changes
def analyze_regional_waste():
    regional_data = pd.DataFrame({
        'Region': ['Gangneung Songjeong', 'Ganghwa Yeochari', 'Geoje Dumo', 'Gochang Dongho', 'Goheung Sinheung'],
        '2008 Count (EA)': [3709, 1824, 1389, 1811, 1442],
        '2012 Count (EA)': [1148, 1779, 803, 1771, 1947]
    })
    regional_data["Change"] = regional_data["2012 Count (EA)"] - regional_data["2008 Count (EA)"]
    return regional_data

# Function to train and evaluate a predictive model for waste
def train_predict_model():
    data = yearly_waste.dropna()
    X = data["Year"].values.reshape(-1, 1)
    y = data["Count (EA)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

# Function to predict future waste data
def predict_future(model):
    future_years = pd.DataFrame({"Year": [2024, 2025, 2026]})
    predictions = model.predict(future_years)
    future_years["Predicted Waste Count (EA)"] = predictions
    return future_years

# Function to analyze algae density trends
def analyze_algae():
    grouped = algae_data.groupby("Year")["Density (g/m2)"].mean()
    grouped.plot(title="Algae Density Over Years", marker="o", figsize=(8, 5))
    plt.show()
    return grouped

# Function to train and predict algae density
def train_predict_algae():
    data = algae_data.dropna()
    X = data["Year"].values.reshape(-1, 1)
    y = data["Density (g/m2)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

# Function to predict future algae density
def predict_future_algae(model):
    future_years = pd.DataFrame({"Year": [2024, 2025, 2026]})
    predictions = model.predict(future_years)
    future_years["Predicted Algae Density (g/m2)"] = predictions
    return future_years

# GUI Interface
class AnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analysis Dashboard")

        # Layout
        frame = ttk.Frame(root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Buttons for each analysis option
        ttk.Button(frame, text="Show Waste EDA", command=self.show_eda).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Analyze Regional Waste", command=self.show_regional_waste).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Train Waste Model", command=self.train_waste_model).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(frame, text="Predict Waste Future", command=self.predict_waste_future).grid(row=0, column=3, padx=5, pady=5)

        ttk.Button(frame, text="Show Algae Trends", command=self.show_algae_trends).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Train Algae Model", command=self.train_algae_model).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Predict Algae Future", command=self.predict_algae_future).grid(row=1, column=2, padx=5, pady=5)

        # Output display
        self.output = tk.Text(frame, wrap="word", width=100, height=30)
        self.output.grid(row=2, column=0, columnspan=4, padx=5, pady=5)

    def show_eda(self):
        self.output.delete(1.0, tk.END)
        grouped = perform_eda()
        self.output.insert(tk.END, "--- Waste EDA ---\n")
        self.output.insert(tk.END, str(grouped))

    def show_regional_waste(self):
        self.output.delete(1.0, tk.END)
        regional_data = analyze_regional_waste()
        self.output.insert(tk.END, "--- Regional Waste Changes ---\n")
        self.output.insert(tk.END, str(regional_data))

    def train_waste_model(self):
        self.output.delete(1.0, tk.END)
        model, mse, r2 = train_predict_model()
        self.waste_model = model
        self.output.insert(tk.END, "--- Waste Model Training Results ---\n")
        self.output.insert(tk.END, f"MSE: {mse:.2f}\nR2: {r2:.2f}\n")

    def predict_waste_future(self):
        if not hasattr(self, "waste_model"):
            self.output.insert(tk.END, "Train the waste model first!\n")
            return

        future_data = predict_future(self.waste_model)
        self.output.insert(tk.END, "--- Future Waste Predictions ---\n")
        self.output.insert(tk.END, str(future_data))

    def show_algae_trends(self):
        self.output.delete(1.0, tk.END)
        grouped = analyze_algae()
        self.output.insert(tk.END, "--- Algae Density Trends ---\n")
        self.output.insert(tk.END, str(grouped))

    def train_algae_model(self):
        self.output.delete(1.0, tk.END)
        model, mse, r2 = train_predict_algae()
        self.algae_model = model
        self.output.insert(tk.END, "--- Algae Model Training Results ---\n")
        self.output.insert(tk.END, f"MSE: {mse:.2f}\nR2: {r2:.2f}\n")

    def predict_algae_future(self):
        if not hasattr(self, "algae_model"):
            self.output.insert(tk.END, "Train the algae model first!\n")
            return

        future_data = predict_future_algae(self.algae_model)
        self.output.insert(tk.END, "--- Future Algae Predictions ---\n")
        self.output.insert(tk.END, str(future_data))

# Main function
def main():
    root = tk.Tk()
    app = AnalysisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
