import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

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
        
    def load_data(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            try:
                self.dataset = pd.read_csv(self.file_path)
                messagebox.showinfo("Success", "Dataset loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset:\n{e}")
    
    def predict(self):
        if self.dataset is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        
        try:
            features = [float(entry.get()) for entry in self.input_fields]
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter numeric values for features.")
            return
        
        if self.dataset.isnull().values.any():
            messagebox.showerror("Error", "Dataset contains missing values. Please preprocess the data.")
            return
        
        x = self.dataset[['smokes', 'anxiety', 'wheezing', 'peerpressure', 'alcoholconsumption', 'yellowfingers']]
        y = self.dataset['result']
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)
        
        sc_x = StandardScaler()
        x_train_scaled = sc_x.fit_transform(x_train)
        x_test_scaled = sc_x.transform(x_test)
        
        classifier = DecisionTreeClassifier()
        classifier.fit(x_train_scaled, y_train)
        
        prediction = classifier.predict(sc_x.transform([features]))
        
        self.result_label.config(text=f"Predicted Lung Cancer: {'Positive' if prediction[0] == 1 else 'Negative'}")
        
        y_pred = classifier.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:\n", report)

if __name__ == "__main__":
    root = tk.Tk()
    app = LungCancerPredictionApp(root)
    root.mainloop()
