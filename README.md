# Diabetes Prediction — Model Comparison (KNN vs Naive Bayes vs SVM)
A machine learning project that trains and compares three classification algorithms on the same dataset to see which performs best at predicting diabetes.

📌 Overview 

Rather than relying on a single algorithm, this project trains K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine on the Pima Indians Diabetes dataset and evaluates them side by side using the same train/test split and scaling pipeline.

🛠️ Tech Stack
* Python
* Pandas, NumPy
* Scikit-learn (KNeighborsClassifier, GaussianNB, SVC, StandardScaler, train_test_split)

🔍 Workflow
* Loaded and explored the diabetes dataset (768 rows, 9 columns)
* Checked class balance (500 non-diabetic, 268 diabetic), missing values, and duplicates
* Split data into training and testing sets (70/30)
* Scaled features using StandardScaler — fitted on training data only
* Trained all three classifiers in a single loop for a fair comparison
* Evaluated each with accuracy, precision, recall, and F1-score

📊 Results
* Model	Accuracy	Recall (diabetic class)
* Gaussian Naive Bayes	76.19%	0.51
* SVM (SVC)	75.76%	0.49
* KNN (k=7)	74.46%	0.51

Key takeaway: Gaussian Naive Bayes edged out the others on accuracy, but all three models struggled with recall on the diabetic class (~0.5), meaning roughly half of actual diabetic cases were missed. In a healthcare context, recall matters more than raw accuracy — this points to class imbalance as the next problem to address.

📁 Dataset

Pima Indians Diabetes dataset — features include Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.

🚀 How to Run
* Clone this repository
* Install dependencies: pip install numpy pandas scikit-learn
* Open the notebook and run all cells

🔮 Possible Improvements
* Handle class imbalance with SMOTE or class weights
* Tune hyperparameters (k value, SVM kernel and C) using GridSearchCV
* Treat zero values in Glucose, BloodPressure, and BMI as missing and impute them
