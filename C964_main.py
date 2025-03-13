import sys
import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pandas.plotting import scatter_matrix

class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)  # Insert message at end of text
        self.text_widget.see(tk.END)  # Scroll to the bottom

    def flush(self):
        pass

def LogisticRegression51():

    df = pd.read_csv("dataset.csv")             # Loads dataset

    df = df.dropna(subset=['Speed_of_Impact'])  # Removes rows where Speed_of_Impact is NaN

    df = df[df['Seatbelt_Used'] == "No"]        # Strip data where seatbelt was used.
    df = df[df['Helmet_Used'] == "No"]          # Strip data where helmet was used.

    X = df[['Speed_of_Impact']]                 # Predictor variable
    y = df['Survived']                          # Target variable (0 = died, 1 = survived)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)      # Splits the dataset into training and testing sets (80% train, 20% test)

    model = LogisticRegression()                # Initializes and trains the model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)   # validate based on results and predictions
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")    # print accuracy info
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    beta_0 = model.intercept_[0]                # Find threshold speed for 51% survival probability
    beta_1 = model.coef_[0][0]

    threshold_speed_51 = (np.log(0.51 / (1 - 0.51)) - beta_0) / beta_1      # Find 51% survive threshhold
    print(f"Estimated Speed Threshold for 51% Survival Probability: {threshold_speed_51:.2f} km/h")

    X_range = np.linspace(X.min(), X.max(), 100)       # Visualizing log curve
    y_prob = model.predict_proba(X_range)[:, 1]             # Probability of survival

    plt.scatter(X, y, color='blue', label="Actual Data")    # plt visual creation
    plt.plot(X_range, y_prob, color='red', linewidth=2, label="Logistic Regression Curve")
    plt.axvline(x=threshold_speed_51, color='green', linestyle='--', label="51% Survival Threshold")
    plt.xlabel("Car Crash Speed")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.show()

def LogisticRegressionGroupings():

    df = pd.read_csv("dataset.csv")  # Loads dataset

    df = df.dropna(subset=['Speed_of_Impact'])  # Removes rows where Speed_of_Impact is NaN

    df = df[df['Seatbelt_Used'] == "No"]  # Strip data where seatbelt was used.
    df = df[df['Helmet_Used'] == "No"]  # Strip data where helmet was used.

    X = df[['Speed_of_Impact']]  # Predictor variable
    y = df['Survived']  # Target variable (0 = died, 1 = survived)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)  # Splits the dataset into training and testing sets (80% train, 20% test)

    model = LogisticRegression()  # Initializes and trains the model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)  # validate based on results and predictions
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")  # print accuracy info
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    beta_0 = model.intercept_[0]  # Find threshold speed for 51% survival probability
    beta_1 = model.coef_[0][0]

    threshold_speed_61 = (np.log(0.61 / (1 - 0.61)) - beta_0) / beta_1          # Solve for survivability thresholds
    print(f"Estimated Speed Threshold for 61% Survival Probability: {threshold_speed_61:.2f} km/h")
    threshold_speed_51 = (np.log(0.51 / (1 - 0.51)) - beta_0) / beta_1
    print(f"Estimated Speed Threshold for 51% Survival Probability: {threshold_speed_51:.2f} km/h")
    threshold_speed_41 = (np.log(0.41 / (1 - 0.41)) - beta_0) / beta_1
    print(f"Estimated Speed Threshold for 41% Survival Probability: {threshold_speed_41:.2f} km/h")
    threshold_speed_31 = (np.log(0.31 / (1 - 0.31)) - beta_0) / beta_1
    print(f"Estimated Speed Threshold for 31% Survival Probability: {threshold_speed_31:.2f} km/h")
    threshold_speed_21 = (np.log(0.21 / (1 - 0.21)) - beta_0) / beta_1
    print(f"Estimated Speed Threshold for 21% Survival Probability: {threshold_speed_21:.2f} km/h")

    X_range = np.linspace(X.min(), X.max(), 100)            # Visualizing log curve
    y_prob = model.predict_proba(X_range)[:, 1]                  # Probability of survival

    plt.scatter(X, y, color='blue', label="Actual Data")        # plt creating visuals
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
    df = pd.read_csv("dataset.csv",names = names)       # create Dataframe

    numeric_df = df.apply(pd.to_numeric, errors='coerce') # sets columns to numeric
    numeric_df = numeric_df.dropna(axis=1, how='all')     # drop NaN columns

    if numeric_df.shape[1] == 0:                          # exception catch
        print("Error: Still no numeric columns after conversion")
        return

    scatter_matrix(numeric_df, figsize=(6, 6), diagonal='hist')     # creates scatterplot visuals
    plt.show()

def Histogram():

    df = pd.read_csv("dataset.csv")

    bins = [0, 20, 40, 60, 80, 100, 120]  # Speed bins
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "100-120"]  # Labels for bins

    df['Speed_Group'] = pd.cut(df['Speed_of_Impact'], bins=bins, labels=labels, right=False)     # Assign speed groups to new column


    survival_rates = df.groupby('Speed_Group')['Survived'].mean() * 100     # Calculate survive % per speed grp

    plt.figure(figsize=(8, 5)) # Histogram plotting
    plt.bar(survival_rates.index, survival_rates, width=0.8, color='skyblue', edgecolor='black')

    plt.xlabel("Speed Group (mph)")
    plt.ylabel("Survivability Percentage (%)")
    plt.title("Survivability Percentage by Speed Group")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()



window = tk.Tk()            # Main window for program
window.title("C964 Traffic Safety Analysis Program")
window.configure(background="white")
window.minsize(400, 400)
window.maxsize(800, 800)

label = tk.Label(window, text="Welcome, please click a button to see the crash speed survivability data.")
label.pack()
label = tk.Label(window, text="To generate new results, please place a csv file named 'dataset.csv' into the same folder.")
label.pack()

console_output = tk.Text(window, height=25, width=120)
console_output.pack(pady=10, padx=10)

sys.stdout = ConsoleRedirector(console_output)

button = tk.Button(window, text="LogisticRegression 51%", command=LogisticRegression51)
button.pack()
button = tk.Button(window, text="LogisticRegression Speed Groupings", command=LogisticRegressionGroupings)
button.pack()
button = tk.Button(window, text="Unfiltered Data Scatterplot", command=Scatterplot)
button.pack()
button = tk.Button(window, text="Histogram", command=Histogram)
button.pack()
button = tk.Button(window, text="Exit", command=window.destroy)
button.pack()

window.mainloop()       # Loops program until exit