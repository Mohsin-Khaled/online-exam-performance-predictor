# Online Exam Performance Predictor 🎓📊

This project is a machine learning–based web application that predicts a student's exam performance (Pass/Fail) based on academic and behavioral factors such as study hours, sleep hours, attendance, internet usage, and internal score.

The application is built using Python, Machine Learning (Random Forest), and Streamlit for an interactive user interface.

---

## 🔍 Features

- Predicts exam result (Pass / Fail) for individual students
- Displays detailed student information
- Interactive visualizations:
  - Study Hours vs Score comparison chart
  - Performance analysis charts
- Internet usage level indicator
- Clean and user-friendly Streamlit UI

---

## 🧠 Machine Learning Model

- Algorithm Used: *Random Forest Classifier*
- Trained using historical student performance data
- Encoded categorical features using Label Encoder
- Model and encoder saved using Pickle (model.pkl, label_encoder.pkl)

---

## 🗂 Dataset Attributes

- Roll_No  
- Name  
- Hours_Studied  
- Sleep_Hours  
- Attendance  
- Internet_Usage  
- Score  
- Result (Target Variable)

---

## 🛠 Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Matplotlib / Seaborn  

---

## 🚀 How to Run the Project Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
