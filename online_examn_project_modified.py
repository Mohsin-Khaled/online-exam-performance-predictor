import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("online_exam_performance_data.csv")

# Encode the categorical column 'Internet_Usage'
le = LabelEncoder()
df['Internet_Usage'] = le.fit_transform(df['Internet_Usage'])

# ✅ Remove columns that should not be used for training
# (Roll_No and Result are not features)
X = df[['Hours_Studied', 'Sleep_Hours', 'Attendance', 'Internet_Usage', 'Score']]
y = df['Result']   # Target variable (Pass / Fail)

# Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the Random Forest model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Save trained model and encoder
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model and Label Encoder have been successfully trained and saved!")
