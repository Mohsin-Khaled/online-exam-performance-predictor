import streamlit as st
import pandas as pd
import pickle
import altair as alt

# ------------------------------------------------------------
# Load Dataset and Model
# ------------------------------------------------------------
df = pd.read_csv("online_exam_performance_data.csv")

# Load trained model and label encoder
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

st.title("📊 Online Exam Performance Predictor")

# ------------------------------------------------------------
# User Input Section
# ------------------------------------------------------------
roll_no = st.text_input("🔢 Enter Student Roll No(1-50):")

# ------------------------------------------------------------
# When user enters a valid roll number
# ------------------------------------------------------------
if roll_no:
    try:
        roll_no = int(roll_no)
        student_data = df[df['Roll_No'] == roll_no]

        if student_data.empty:
            st.warning("⚠️ No student found with that Roll No.")
        else:
            st.success(f"Showing performance for Roll No: {roll_no}")

            # Display basic student details
            st.write("### 🎓 Student Details")
            st.dataframe(student_data[['Roll_No','Name', 'Hours_Studied', 'Sleep_Hours', 'Attendance', 'Internet_Usage', 'Score']])

            # Prepare input for prediction (drop Result column)
            features = student_data[['Hours_Studied', 'Sleep_Hours', 'Attendance', 'Internet_Usage', 'Score']].copy()

            # Encode Internet_Usage to match the trained model
            features['Internet_Usage'] = le.transform(features['Internet_Usage'])

            # Predict result dynamically
            prediction = model.predict(features)
            result_text = "✅ PASS" if prediction[0] == 1 else "❌ FAIL"
            result_color = "#00cc66" if prediction[0] == 1 else "#ff4d4d"

            # Display Predicted Result Box with Border
            st.markdown(f"""
                <div style="
                    background-color:#1a1a1a;
                    border: 3px solid {result_color};
                    border-radius: 20px;
                    padding: 15px;
                    margin-top: 10px;
                    box-shadow: 0px 0px 10px {result_color}55;
                ">
                    <h4 style='color:white;'>📈 Predicted Exam Result</h4>
                    <h3 style='color:{result_color}; font-size:24px; font-weight:bold;'>{result_text}</h3>
                </div>
            """, unsafe_allow_html=True)

            # Internet Usage Widget with Border
            internet_usage_val = student_data['Internet_Usage'].values[0]

            st.markdown(f"""
                <div style="
                    background-color:#1a1a1a;
                    border: 3px solid #1E90FF;
                    border-radius: 20px;
                    padding: 15px;
                    margin-top: 15px;
                    box-shadow: 0px 0px 10px rgba(30,144,255,0.5);
                ">
                    <h4 style='color:white;'>🌐 Internet Usage Level</h4>
                    <h3 style='color:#1E90FF; font-size:22px; font-weight:bold;'>{internet_usage_val}</h3>
                </div>
            """, unsafe_allow_html=True)

            # ------------------------------------------------------------
            # Combined Bar Chart (Hours Studied vs Score)
            # ------------------------------------------------------------
            st.subheader("📊 Study vs Score Comparison")
            comparison_chart = alt.Chart(student_data).transform_fold(
                ['Hours_Studied', 'Score'],
                as_=['Category', 'Value']
            ).mark_bar(size=40).encode(
                x=alt.X('Category:N', title='Category'),
                y=alt.Y('Value:Q', title='Value'),
                color='Category:N'
            ).properties(width=500, height=300)
            st.altair_chart(comparison_chart)

            # ------------------------------------------------------------
            # Pie Chart for Student Category Comparison
            # (Internet_Usage, Attendance, Sleep_Hours)
            # ------------------------------------------------------------
            st.subheader("🥧 Student Category Comparison")

            # Get raw values from the student row
            internet_raw = str(student_data['Internet_Usage'].values[0]).strip().capitalize()
            attendance_val = float(student_data['Attendance'].values[0])
            sleep_val = float(student_data['Sleep_Hours'].values[0])

            # Reliable fallback mapping for Internet_Usage
            fallback_map = {'Low': 1.0, 'Medium': 2.0, 'High': 3.0}

            # Get mapped numeric value (default to 2.0 if missing)
            internet_val = fallback_map.get(internet_raw, 2.0) * 30  # Scaled for better visibility

            # Prepare DataFrame for pie chart
            category_data = pd.DataFrame({
                'Category': ['Internet_Usage', 'Attendance', 'Sleep_Hours'],
                'Value': [internet_val, attendance_val, sleep_val]
            })

            # Draw pie chart
            pie_chart = alt.Chart(category_data).mark_arc(innerRadius=60).encode(
                theta='Value:Q',
                color='Category:N',
                tooltip=['Category', 'Value']
            ).properties(width=400, height=300)

            st.altair_chart(pie_chart)


    except ValueError:
        st.error("⚠️ Please enter a valid numeric Roll Number.")

