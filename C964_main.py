import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tkinter as tk
import sys
from pandas.plotting import scatter_matrix

class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)  # Insert message at the end of text widget
        self.text_widget.see(tk.END)  # Auto-scroll to the bottom

    def flush(self):
        pass  # Required for compatibility with sys.stdout

def LogisticRegression51():

    # Load  dataset
    df = pd.read_csv("dataset.csv")

    df = df.dropna(subset=['Speed_of_Impact'])  # Removes rows where Speed_of_Impact is NaN

    # Filter data to only include rows where Seatbelt_Used/Helmet_used is yes
    df = df[df['Seatbelt_Used'] == "No"]
    df = df[df['Helmet_Used'] == "No"]

    # Select relevant variables
    X = df[['Speed_of_Impact']]  # Predictor variable
    y = df['Survived']  # Target variable (0 = did not survive, 1 = survived)

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Compute the threshold speed for 51% survival probability
    beta_0 = model.intercept_[0]  # Intercept
    beta_1 = model.coef_[0][0]  # Slope

    # Solve for survivability threshold
    threshold_speed_51 = (np.log(0.51 / (1 - 0.51)) - beta_0) / beta_1
    print(f"Estimated Speed Threshold for 51% Survival Probability: {threshold_speed_51:.2f} km/h")

    # Visualizing the logistic regression curve
    X_range = np.linspace(X.min(), X.max(), 100)
    y_prob = model.predict_proba(X_range)[:, 1]  # Probability of survival

    plt.scatter(X, y, color='blue', label="Actual Data")
    plt.plot(X_range, y_prob, color='red', linewidth=2, label="Logistic Regression Curve")
    plt.axvline(x=threshold_speed_51, color='green', linestyle='--', label="51% Survival Threshold")
    plt.xlabel("Car Crash Speed")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.show()

def LogisticRegressionGroupings():

    # Load  dataset
    # df = pd.read_csv(".venv/dataset")
    df = pd.read_csv("dataset.csv")

    # print(df.columns)
    df = df.dropna(subset=['Speed_of_Impact'])  # Removes rows where Speed_of_Impact is NaN

    # Filter data to only include rows where Seatbelt_Used/Helmet_used is yes
    df = df[df['Seatbelt_Used'] == "No"]
    df = df[df['Helmet_Used'] == "No"]

    # Select relevant variables
    X = df[['Speed_of_Impact']]  # Predictor variable
    y = df['Survived']  # Target variable (0 = did not survive, 1 = survived)

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Compute the threshold speed for 51% survival probability
    beta_0 = model.intercept_[0]  # Intercept
    beta_1 = model.coef_[0][0]  # Slope

    # Solve for survivability threshold
    threshold_speed_61 = (np.log(0.61 / (1 - 0.61)) - beta_0) / beta_1
    print(f"Estimated Speed Threshold for 61% Survival Probability: {threshold_speed_61:.2f} km/h")
    threshold_speed_51 = (np.log(0.51 / (1 - 0.51)) - beta_0) / beta_1
    print(f"Estimated Speed Threshold for 51% Survival Probability: {threshold_speed_51:.2f} km/h")
    threshold_speed_41 = (np.log(0.41 / (1 - 0.41)) - beta_0) / beta_1
    print(f"Estimated Speed Threshold for 41% Survival Probability: {threshold_speed_41:.2f} km/h")
    threshold_speed_31 = (np.log(0.31 / (1 - 0.31)) - beta_0) / beta_1
    print(f"Estimated Speed Threshold for 31% Survival Probability: {threshold_speed_31:.2f} km/h")
    threshold_speed_21 = (np.log(0.21 / (1 - 0.21)) - beta_0) / beta_1
    print(f"Estimated Speed Threshold for 21% Survival Probability: {threshold_speed_21:.2f} km/h")

    # Visualizing the logistic regression curve
    X_range = np.linspace(X.min(), X.max(), 100)
    y_prob = model.predict_proba(X_range)[:, 1]  # Probability of survival

    plt.scatter(X, y, color='blue', label="Actual Data")
    plt.plot(X_range, y_prob, color='red', linewidth=2, label="Logistic Regression Curve")
    plt.axvline(x=threshold_speed_61, color='teal', linestyle='--', label="61% Survival Threshold")
    plt.axvline(x=threshold_speed_51, color='green', linestyle='--', label="51% Survival Threshold")
    plt.axvline(x=threshold_speed_41, color='orange', linestyle='--', label="41% Survival Threshold")
    plt.axvline(x=threshold_speed_31, color='purple', linestyle='--', label="31% Survival Threshold")
    plt.axvline(x=threshold_speed_21, color='black', linestyle='--', label="21% Survival Threshold")
    plt.xlabel("Car Crash Speed")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.show()

def Scatterplot():

    names = ['Age','Gender','Speed_of_Impact','Helmet_Used','Seatbelt_Used','Survived']
    df = pd.read_csv("dataset.csv",names = names)

    # Convert all columns to numeric (coercing errors to NaN)
    numeric_df = df.apply(pd.to_numeric, errors='coerce')

    # Drop columns that are fully NaN
    numeric_df = numeric_df.dropna(axis=1, how='all')

    if numeric_df.shape[1] == 0:
        print("Error: Still no numeric columns after conversion")
        return

    # Create scatter plot matrix
    scatter_matrix(numeric_df, figsize=(6, 6), diagonal='hist')
    plt.show()


# Create the main window
window = tk.Tk()
window.title("C964 Traffic Safety Analysis Program")
window.configure(background="white")
window.minsize(400, 400)
window.maxsize(800, 800)
#window.geometry("400x400+100+100")

# Add a label widget
label = tk.Label(window, text="Welcome, please click a button to see the crash speed survivability data.")
label.pack()

console_output = tk.Text(window, height=30, width=120)
console_output.pack(pady=10, padx=10)

sys.stdout = ConsoleRedirector(console_output)

# Add a button widget
button = tk.Button(window, text="LogisticRegression 51%", command=LogisticRegression51)
button.pack()
button = tk.Button(window, text="LogisticRegression Speed Groupings", command=LogisticRegressionGroupings)
button.pack()
button = tk.Button(window, text="Unfiltered Data Scatterplot", command=Scatterplot)
button.pack()
button = tk.Button(window, text="Exit", command=window.destroy)
button.pack()

# Start the Tkinter event loop
window.mainloop()

