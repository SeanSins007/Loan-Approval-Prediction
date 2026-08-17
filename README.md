README.md
Markdown

# 🏦 Loan Approval Prediction

A machine learning web application that predicts whether a loan application
will be **Approved** or **Rejected** based on applicant details.

Built with **Python**, **Flask**, and **Scikit-Learn** as an intermediate-level
portfolio project demonstrating end-to-end ML development.

---

## 📌 Project Overview

This project covers the complete machine learning workflow:

- Loading and exploring a real-world loan dataset
- Preprocessing data with automated pipelines
- Training and comparing multiple classification models
- Selecting the best model automatically based on F1 Score
- Serving predictions through a clean Flask web application
- Displaying results with confidence scores and Bootstrap 5 UI

---

## ✨ Features

- ✅ Automated target column detection
- ✅ Automated feature type detection (numerical vs categorical)
- ✅ Dynamic form generation based on dataset columns
- ✅ Full preprocessing pipeline (imputation, encoding, scaling)
- ✅ Trains and compares 3 classification models
- ✅ Automatically selects the best model by F1 Score
- ✅ Saves model comparison, classification report, confusion matrix
- ✅ Flask web app with Bootstrap 5 responsive UI
- ✅ Prediction confidence score with progress bar
- ✅ Input validation with friendly error messages

---

## 📁 Folder Structure
```
Loan Approval Prediction/
│
│── app.py # Flask web application
│── train.py # Main training script
│── predict.py # Prediction logic
│── preprocessing.py # Preprocessing pipeline
│── config.py # Centralized configuration
│── utils.py # Shared utility functions
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│── .gitignore # Git ignore rules
│
├── data/
│ loan_data.csv # Input dataset (add your own)
│
├── models/
│ best_model.pkl # Saved best model (generated)
│ preprocessor.pkl # Saved preprocessor (generated)
│
├── results/
│ model_comparison.csv # Model metrics comparison
│ classification_report.csv # Best model classification report
│ confusion_matrix.png # Confusion matrix visualization
│ metrics.json # Metrics in JSON format
│
├── src/
│ data_loader.py # Dataset loading and validation
│ feature_engineering.py# Feature preparation and pipelines
│ model_training.py # Model definitions and training
│ evaluation.py # Metrics, reports, and visualizations
│
├── templates/
│ index.html # Main HTML template
│
└── static/
css/
style.css # Custom CSS styles
images/ # Static images folder
```
---

## 🛠️ Technologies Used

| Category        | Technology          |
|----------------|---------------------|
| Language        | Python 3.10+        |
| Web Framework   | Flask 3.0           |
| ML Library      | Scikit-Learn 1.5    |
| Data Processing | Pandas, NumPy       |
| Visualization   | Matplotlib          |
| Model Saving    | Joblib              |
| Frontend        | Bootstrap 5, HTML5  |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Loan_Approval_Prediction.git
cd Loan_Approval_Prediction
2. Create a Virtual Environment
Bash

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
4. Add the Dataset
Place your loan_data.csv file inside the data/ folder:

text

data/
  loan_data.csv
The dataset should contain applicant features and a target column
indicating loan approval status (e.g., loan_status, approved).

🚀 How to Train
Run the training script to preprocess data, train models, and save artifacts:

Bash

python train.py
This will automatically:

Load and preprocess the dataset
Train Logistic Regression, Decision Tree, and Random Forest
Compare models and select the best one by F1 Score
Save models/best_model.pkl and models/preprocessor.pkl
Generate all result files in the results/ folder
Training Output Example
text

============================================================
   Loan Approval Prediction — Training Pipeline
============================================================

[Data Loader] Dataset loaded successfully: 614 rows, 13 columns
[Feature Engineering] Numerical columns : [...]
[Feature Engineering] Categorical columns: [...]
[Model Training] Training: Logistic Regression ...
[Model Training] Training: Decision Tree ...
[Model Training] Training: Random Forest ...
[Model Training] Best model selected: 'Random Forest' (F1 Score: 0.8142)

============================================================
   Training Complete!
============================================================
🌐 How to Run
After training, start the Flask web application:

Bash

python app.py
Then open your browser and go to:

text

http://localhost:5000
Fill in the loan application form and click Check Loan Approval
to get an instant prediction with confidence score.

📊 Results
After training, the following files are generated in results/:

File	Description
model_comparison.csv	Accuracy, Precision, Recall, F1 for all models
classification_report.csv	Detailed report for the best model
confusion_matrix.png	Confusion matrix visualization
metrics.json	Best model metrics in JSON format
🖥️ Application Screenshots
Add screenshots of your running application here after deployment.

🔮 Future Improvements
 Add more classification models (XGBoost, SVM)
 Add hyperparameter tuning with GridSearchCV
 Add SHAP values for model explainability
 Add a results dashboard with charts
 Add REST API endpoints for external integrations
 Deploy to cloud (Render, Railway, or AWS)
 Add user authentication for saved predictions
 Support CSV file upload for batch predictions
📄 License
This project is licensed under the MIT License.

text

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
🙌 Acknowledgements
Scikit-Learn Documentation
Flask Documentation
Bootstrap 5
Kaggle Loan Prediction Dataset
