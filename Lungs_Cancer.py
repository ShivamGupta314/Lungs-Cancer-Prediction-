import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class LungCancerPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lung Cancer Prediction")
        self.root.geometry("500x600")
        
       
        self.root.configure(bg="#f0f0f0")
        
       
        self.background_image = Image.open("/Users/shivamgupta/Desktop/Projects/machine_learning_project/lungcancer_1280.jpg")
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        self.background_label = tk.Label(self.root, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.file_path = None
        self.dataset = None
        
        
        self.header_label = tk.Label(self.root, text="Lung Cancer Prediction", font=("Helvetica", 24, "bold"), bg="#f0f0f0", fg="#333")
        self.header_label.pack(pady=20)
        
        # Load Dataset Button
        self.load_data_button = tk.Button(self.root, text="Load Dataset", command=self.load_data, bg="#008CBA", fg="black", font=("Helvetica", 14, "bold"))
        self.load_data_button.pack(pady=10, padx=20, ipadx=20)
        
       
        self.input_fields = []
        labels = ["Smokes (1 for Yes, 0 for No)", "Anxiety (1 for Yes, 0 for No)", "Wheezing (1 for Yes, 0 for No)", "Peer Pressure (1 for Yes, 0 for No)", "Alcohol Consumption (1 for Yes, 0 for No)", "Yellow Fingers (1 for Yes, 0 for No)"]
        for label_text in labels:
            label = tk.Label(self.root, text=label_text, bg="#f0f0f0", fg="#333", font=("Helvetica", 12))
            label.pack()
            entry = tk.Entry(self.root, font=("Helvetica", 12))
            entry.pack(pady=5)
            self.input_fields.append(entry)
        
       
        self.predict_button = tk.Button(self.root, text="Predict Lung Cancer", command=self.predict, bg="#008CBA", fg="black", font=("Helvetica", 14, "bold"))
        self.predict_button.pack(pady=20, padx=20, ipadx=20)
        
       
        self.result_label = tk.Label(self.root, text="", bg="#f0f0f0", fg="#333", font=("Helvetica", 16, "bold"))
        self.result_label.pack(pady=5)
        
        # Define model file paths
        self.dt_model_path = 'decision_tree_model.joblib'
        self.lr_model_path = 'logistic_regression_model.joblib'
        self.rf_model_path = 'random_forest_model.joblib'
        
    def load_data(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            try:
                self.dataset = pd.read_csv(self.file_path)
                # Fill missing values with the mode of each column
                for column in self.dataset.columns:
                    self.dataset[column].fillna(self.dataset[column].mode()[0], inplace=True)
                messagebox.showinfo("Success", "Dataset loaded successfully and missing values handled.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset:\n{e}")
    
    def predict(self):
        if self.dataset is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        
        # Validate and collect input features
        features = []
        try:
            for i, entry in enumerate(self.input_fields):
                value = entry.get()
                if value not in ['0', '1']:
                    messagebox.showerror("Error", f"Invalid input for {labels[i]}. Please enter only 0 or 1.")
                    return
                features.append(float(value))
        except ValueError:
             # This part might be redundant with the '0', '1' check, but keep for robustness
            messagebox.showerror("Error", "Invalid input. Please enter numeric values (0 or 1) for features.")
            return
        
        x = self.dataset[['smokes', 'anxiety', 'wheezing', 'peerpressure', 'alcoholconsumption', 'yellowfingers']]
        y = self.dataset['result']
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)
        
        sc_x = StandardScaler()
        x_train_scaled = sc_x.fit_transform(x_train)
        x_test_scaled = sc_x.transform(x_test)
        
        best_dt_classifier = None
        best_lr_classifier = None
        best_rf_classifier = None
        
        # Check if models are already saved
        if os.path.exists(self.dt_model_path) and os.path.exists(self.lr_model_path) and os.path.exists(self.rf_model_path):
            print("\nLoading saved models...")
            best_dt_classifier = joblib.load(self.dt_model_path)
            best_lr_classifier = joblib.load(self.lr_model_path)
            best_rf_classifier = joblib.load(self.rf_model_path)
            print("Models loaded successfully.")
            
            # Evaluate loaded models
            y_pred_dt = best_dt_classifier.predict(x_test_scaled)
            accuracy_dt = accuracy_score(y_test, y_pred_dt)
            report_dt = classification_report(y_test, y_pred_dt)
            print("\n--- Decision Tree Evaluation (Loaded Model) ---")
            print("Accuracy:", accuracy_dt)
            print("Classification Report:\n", report_dt)
            
            y_pred_lr = best_lr_classifier.predict(x_test_scaled)
            accuracy_lr = accuracy_score(y_test, y_pred_lr)
            report_lr = classification_report(y_test, y_pred_lr)
            print("\n--- Logistic Regression Evaluation (Loaded Model) ---")
            print("Accuracy:", accuracy_lr)
            print("Classification Report:\n", report_lr)
            
            y_pred_rf = best_rf_classifier.predict(x_test_scaled)
            accuracy_rf = accuracy_score(y_test, y_pred_rf)
            report_rf = classification_report(y_test, y_pred_rf)
            print("\n--- Random Forest Evaluation (Loaded Model) ---")
            print("Accuracy:", accuracy_rf)
            print("Classification Report:\n", report_rf)
            
        else:
            print("\nTraining and saving new models...")
            # Hyperparameter tuning for Decision Tree
            dt_param_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=0), dt_param_grid, cv=5)
            dt_grid_search.fit(x_train_scaled, y_train)
            
            best_dt_classifier = dt_grid_search.best_estimator_
            
            # Hyperparameter tuning for Logistic Regression
            lr_param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            }
            lr_grid_search = GridSearchCV(LogisticRegression(random_state=0), lr_param_grid, cv=5)
            lr_grid_search.fit(x_train_scaled, y_train)
            
            best_lr_classifier = lr_grid_search.best_estimator_
            
            # Hyperparameter tuning for Random Forest
            rf_param_grid = {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=0), rf_param_grid, cv=5)
            rf_grid_search.fit(x_train_scaled, y_train)
            
            best_rf_classifier = rf_grid_search.best_estimator_
            
            # Save the best models
            joblib.dump(best_dt_classifier, self.dt_model_path)
            joblib.dump(best_lr_classifier, self.lr_model_path)
            joblib.dump(best_rf_classifier, self.rf_model_path)
            print("Models trained and saved successfully.")
            
            # Evaluate newly trained models
            y_pred_dt = best_dt_classifier.predict(x_test_scaled)
            accuracy_dt = accuracy_score(y_test, y_pred_dt)
            report_dt = classification_report(y_test, y_pred_dt)
            print("\n--- Decision Tree Evaluation (Newly Trained Model) ---")
            print("Best Parameters:", dt_grid_search.best_params_)
            print("Accuracy:", accuracy_dt)
            print("Classification Report:\n", report_dt)
            
            y_pred_lr = best_lr_classifier.predict(x_test_scaled)
            accuracy_lr = accuracy_score(y_test, y_pred_lr)
            report_lr = classification_report(y_test, y_pred_lr)
            print("\n--- Logistic Regression Evaluation (Newly Trained Model) ---")
            print("Best Parameters:", lr_grid_search.best_params_)
            print("Accuracy:", accuracy_lr)
            print("Classification Report:\n", report_lr)
            
            y_pred_rf = best_rf_classifier.predict(x_test_scaled)
            accuracy_rf = accuracy_score(y_test, y_pred_rf)
            report_rf = classification_report(y_test, y_pred_rf)
            print("\n--- Random Forest Evaluation (Newly Trained Model) ---")
            print("Best Parameters:", rf_grid_search.best_params_)
            print("Accuracy:", accuracy_rf)
            print("Classification Report:\n", report_rf)
            
        # Make prediction using the best Random Forest model for the UI
        prediction_rf = best_rf_classifier.predict(sc_x.transform([features]))
        print(f"Input features for prediction: {features}")
        print(f"Random Forest prediction result: {prediction_rf[0]}")
        self.result_label.config(text=f"Predicted Lung Cancer (Random Forest): {'Positive' if prediction_rf[0] == 1 else 'Negative'}")
        
        # --- Cross-Validation Evaluation ---
        print("\n--- Cross-Validation Evaluation ---")
        
        # Cross-validation for Decision Tree
        dt_cv_scores = cross_val_score(best_dt_classifier, sc_x.transform(x), y, cv=5)
        print(f"Decision Tree Mean Cross-Validation Accuracy (5-fold): {dt_cv_scores.mean():.4f}")
        
        # Cross-validation for Logistic Regression
        lr_cv_scores = cross_val_score(best_lr_classifier, sc_x.transform(x), y, cv=5)
        print(f"Logistic Regression Mean Cross-Validation Accuracy (5-fold): {lr_cv_scores.mean():.4f}")
        
        # Cross-validation for Random Forest
        rf_cv_scores = cross_val_score(best_rf_classifier, sc_x.transform(x), y, cv=5)
        print(f"Random Forest Mean Cross-Validation Accuracy (5-fold): {rf_cv_scores.mean():.4f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LungCancerPredictionApp(root)
    root.mainloop()
