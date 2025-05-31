# 🩺 Lung Cancer Prediction System (Python + Tkinter + Machine Learning)

A fully beginner-friendly GUI-based machine learning project to predict lung cancer based on common health indicators like smoking, anxiety, etc.

This README includes everything — setup, dataset format, full code explanation **line-by-line inside the README** itself.

---

## 🎯 Objective

- Load a CSV dataset
- User enters 6 binary health-related inputs (0/1)
- Predict lung cancer using ML models (Decision Tree, Logistic Regression, Random Forest)
- Show result in a clean GUI
- Display model accuracy and evaluation in terminal

---

## 🛠️ Setup Instructions

1. ✅ Install required libraries:
```bash
/usr/bin/python3 -m pip install --user pandas scikit-learn Pillow joblib
```

2. ✅ Run the app:
```bash
/usr/bin/python3 /path/to/Lungs_Cancer.py
```

> Replace `/path/to/...` with your actual script location.

---

## 📂 Dataset Format (CSV)

Make sure your dataset has **only these 7 columns**, all values must be 0 or 1 (binary):

| Column              | Meaning                                     |
|---------------------|---------------------------------------------|
| `smokes`            | 1 if person smokes, else 0                  |
| `anxiety`           | 1 if person has anxiety, else 0             |
| `wheezing`          | 1 if person wheezes, else 0                 |
| `peerpressure`      | 1 if peer pressure present, else 0          |
| `alcoholconsumption`| 1 if drinks alcohol, else 0                 |
| `yellowfingers`     | 1 if fingers are yellow, else 0             |
| `result`            | 1 if lung cancer present, else 0 (target)   |

---

## 🧠 Code Explanation (Everything Inline)

### 🔹 Importing Required Libraries

```python
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
```
📝 These libraries are used for:
- GUI creation (`tkinter`)
- CSV data handling (`pandas`)
- Image background (`PIL`)
- Machine learning modeling (`scikit-learn`)
- Saving/loading models (`joblib`)
- File existence check (`os`)

---

### 🔹 GUI Initialization

```python
class LungCancerPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lung Cancer Prediction")
        self.root.geometry("500x600")
```
📝 Sets up the GUI window title and size.

```python
        self.background_image = Image.open("lungcancer_1280.jpg")
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        self.background_label = tk.Label(self.root, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
```
📝 Loads and displays the background image.

```python
        self.load_data_button = tk.Button(self.root, text="Load Dataset", command=self.load_data)
        self.load_data_button.pack(pady=10)
```
📝 Button to choose CSV file from system.

```python
        self.input_fields = []
        labels = ["Smokes", "Anxiety", "Wheezing", "Peer Pressure", "Alcohol Consumption", "Yellow Fingers"]
        for label_text in labels:
            label = tk.Label(self.root, text=label_text)
            label.pack()
            entry = tk.Entry(self.root)
            entry.pack()
            self.input_fields.append(entry)
```
📝 Creates labels and text inputs for all 6 binary inputs.

```python
        self.predict_button = tk.Button(self.root, text="Predict Lung Cancer", command=self.predict)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(self.root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

        self.dt_model_path = 'decision_tree_model.joblib'
        self.lr_model_path = 'logistic_regression_model.joblib'
        self.rf_model_path = 'random_forest_model.joblib'
```
📝 Predict button + result label + model path setup.

---

### 🔹 Load Dataset

```python
    def load_data(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            try:
                self.dataset = pd.read_csv(self.file_path)
                for column in self.dataset.columns:
                    self.dataset[column].fillna(self.dataset[column].mode()[0], inplace=True)
                messagebox.showinfo("Success", "Dataset loaded and cleaned.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
```
📝 Opens file dialog, loads CSV, fills missing values with most frequent value.

---

### 🔹 Prediction Logic

```python
    def predict(self):
        if self.dataset is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
```
📝 If no dataset is loaded, show error.

```python
        features = []
        try:
            for entry in self.input_fields:
                value = entry.get()
                if value not in ['0', '1']:
                    raise ValueError("Only 0 or 1 allowed")
                features.append(float(value))
        except:
            messagebox.showerror("Error", "Invalid input")
            return
```
📝 Ensures all inputs are either 0 or 1.

```python
        x = self.dataset[['smokes', 'anxiety', 'wheezing', 'peerpressure', 'alcoholconsumption', 'yellowfingers']]
        y = self.dataset['result']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        sc_x = StandardScaler()
        x_train_scaled = sc_x.fit_transform(x_train)
        x_test_scaled = sc_x.transform(x_test)
```
📝 Prepares dataset, splits and scales it.

---

### 🔹 Train or Load Models

```python
        if os.path.exists(self.dt_model_path) and os.path.exists(self.lr_model_path) and os.path.exists(self.rf_model_path):
            best_dt_classifier = joblib.load(self.dt_model_path)
            best_lr_classifier = joblib.load(self.lr_model_path)
            best_rf_classifier = joblib.load(self.rf_model_path)
```
📝 Loads models if already saved.

```python
        else:
            dt_grid_search = GridSearchCV(DecisionTreeClassifier(), {'max_depth': [None, 5, 10]}, cv=5)
            dt_grid_search.fit(x_train_scaled, y_train)
            best_dt_classifier = dt_grid_search.best_estimator_

            lr_grid_search = GridSearchCV(LogisticRegression(), {'C': [0.1, 1, 10]}, cv=5)
            lr_grid_search.fit(x_train_scaled, y_train)
            best_lr_classifier = lr_grid_search.best_estimator_

            rf_grid_search = GridSearchCV(RandomForestClassifier(), {'n_estimators': [10, 50, 100]}, cv=5)
            rf_grid_search.fit(x_train_scaled, y_train)
            best_rf_classifier = rf_grid_search.best_estimator_

            joblib.dump(best_dt_classifier, self.dt_model_path)
            joblib.dump(best_lr_classifier, self.lr_model_path)
            joblib.dump(best_rf_classifier, self.rf_model_path)
```
📝 If models don’t exist, trains them and saves them using GridSearch.

---

### 🔹 Make Prediction and Show Result

```python
        prediction_rf = best_rf_classifier.predict(sc_x.transform([features]))
        self.result_label.config(text=f"Predicted: {'Positive' if prediction_rf[0] == 1 else 'Negative'}")
```
📝 Predicts using Random Forest and updates the result on screen.

---

### 🔹 Show Accuracy & Metrics in Terminal

```python
        for model, name in zip([best_dt_classifier, best_lr_classifier, best_rf_classifier], ["Decision Tree", "Logistic Regression", "Random Forest"]):
            print(f"\n{name} Accuracy: {accuracy_score(y_test, model.predict(x_test_scaled))}")
            print(classification_report(y_test, model.predict(x_test_scaled)))
            print(f"Cross-Validation Score: {cross_val_score(model, sc_x.transform(x), y, cv=5).mean()}")
```
📝 Shows how good each model performed.

---

### 🔹 Main Entry Point

```python
if __name__ == "__main__":
    root = tk.Tk()
    app = LungCancerPredictionApp(root)
    root.mainloop()
```
📝 Starts the GUI application.

---

## ✅ Sample Output

**GUI:**  
```
Predicted: Positive
```

**Terminal:**
```
Random Forest Accuracy: 0.92
Classification Report: ...
Cross-Validation Score: 0.91
```

---

## 🔮 Future Improvements

- Export prediction report as PDF
- Add charts and graphs
- Store prediction history in a database
- Package as a standalone app using PyInstaller

---

## 👨‍💻 Author

**Shivam Gupta**  
.NET Developer @ Cognizant | CS Final Year | ML Enthusiast

---

## 📬 Contact

Feel free to connect via GitHub or email.
