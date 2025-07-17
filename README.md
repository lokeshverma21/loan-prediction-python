# 🧠 Loan Approval Prediction using Machine Learning

This project aims to automate the loan approval process using a machine learning model trained on historical loan data. It helps financial institutions make faster and more consistent loan approval decisions.

## 📌 Problem Statement

Banks and financial institutions often face challenges in identifying potential loan defaulters. This project solves that by predicting whether a loan should be approved based on applicant data such as income, credit history, education, and employment status.

## 💡 Technologies Used

- Python 3.x
- Pandas, NumPy, Scikit-learn
- Random Forest Classifier
- Streamlit (for frontend)
- Joblib (for saving model)

## 📂 Project Structure

```
loan_approval_prediction/
├── model/
│   ├── train_model.py          # Script to train and save the ML model
│   └── train_model.ipynb       # Jupyter notebook version (optional)
├── app/
│   ├── app.py                  # Streamlit app for making predictions
│   ├── loan_model.pkl          # Saved ML model
│   └── label_encoders.pkl      # Saved encoders for categorical features
├── data/
│   └── sample_input.csv        # Example test input
├── README.md                   # Project documentation
├── requirements.txt            # Required Python libraries
```

## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/lokeshverma21/loan-prediction-python.git
cd loan-approval-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
cd model
python train_model.py
```

### 4. Run the Streamlit app
```bash
cd ../app
streamlit run app.py
```

### 5. Use the web interface
Enter details such as gender, marital status, credit history, income, and the app will predict whether the loan is likely to be **Approved ✅** or **Not Approved ❌**.

---

## 📊 Example

**Input:**  
- Gender: Male  
- Married: Yes  
- Education: Graduate  
- Credit History: 1  
- Loan Amount: 150

**Output:**  
✅ Loan Approved

---

## 📈 Accuracy

The Random Forest model achieves approximately **81% accuracy** on the test set.

---

## 📌 License

This project is free to use for educational purposes. Attribution appreciated!

---

## 🔗 Dataset Reference

[Loan Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
