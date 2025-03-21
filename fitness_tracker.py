import streamlit as st
import numpy as np
import pandas as pd
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

# Streamlit App Title
st.write("## Personal Fitness Tracker")
st.write(
    "In this WebApp, you can observe your predicted calories burned based on your parameters such as `Age`, `Gender`, `BMI`, etc."
)

# Sidebar for User Input
st.sidebar.header("User Input Parameters:")


def user_input_features():
    age = st.sidebar.slider("Age:", 10, 100, 30)
    bmi = st.sidebar.slider("BMI:", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min):", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate:", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C):", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender:", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # 1 for male, 0 for female
    }

    return pd.DataFrame(data_model, index=[0])


df = user_input_features()

st.write("---")
st.header("Your Parameters:")
st.write(df)

# Check if datasets exist
if not os.path.exists("calories.csv") or not os.path.exists("exercise.csv"):
    st.error("Error: Dataset files are missing!")
    st.stop()

# Load datasets
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

# Merge datasets on User_ID
exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

# Add BMI column before splitting
exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
exercise_df["BMI"] = round(exercise_df["BMI"], 2)

# Train-test split
train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Select features
features = ["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
train_data = train_data[features]
test_data = test_data[features]

# Convert categorical variables
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

# x (features) and y (labels)
X_train, y_train = train_data.drop("Calories", axis=1), train_data["Calories"]
X_test, y_test = test_data.drop("Calories", axis=1), test_data["Calories"]

# Model training
random_reg = RandomForestRegressor(n_estimators=200, max_features=3, max_depth=6, random_state=42)
random_reg.fit(X_train, y_train)

# Align user input with training data columns
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df)

# Display prediction
st.write("---")
st.header("Prediction:")
with st.spinner("Processing..."):
    time.sleep(2)

st.write(f"**{round(prediction[0], 2)} kilocalories**")

# Similar results
st.write("---")
st.header("Similar Results:")

calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (
        exercise_df["Calories"] <= calorie_range[1])]

if similar_data.shape[0] > 5:
    st.write(similar_data.sample(5))
else:
    st.write(similar_data)

# Statistics
st.write("---")
st.header("General Information:")

boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.write(f"You are older than **{round(sum(boolean_age) / len(boolean_age), 2) * 100}%** of users.")
st.write(
    f"Your exercise duration is higher than **{round(sum(boolean_duration) / len(boolean_duration), 2) * 100}%** of users."
)
st.write(
    f"You have a higher heart rate than **{round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100}%** of users."
)
st.write(
    f"You have a higher body temperature than **{round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100}%** of users."
)

# Store input data
if 'user_input_history' not in st.session_state:
    st.session_state.user_input_history = pd.DataFrame(columns=df.columns)

st.session_state.user_input_history = pd.concat([st.session_state.user_input_history, df], ignore_index=True)

# Display input history
st.write("---")
st.header("Your Input History:")
if not st.session_state.user_input_history.empty:
    st.write(st.session_state.user_input_history)
else:
    st.write("No input history available.")

# Future Comparison
st.write("---")
st.header("Comparison with Previous Inputs:")

if len(st.session_state.user_input_history) > 1:
    last_input = st.session_state.user_input_history.iloc[-1]
    previous_inputs = st.session_state.user_input_history.iloc[:-1]

    comparison_results = []
    for index, previous_input in previous_inputs.iterrows():
        diff = last_input - previous_input
        comparison_results.append(diff)

    comparison_df = pd.DataFrame(comparison_results)
    st.write(comparison_df)
else:
    st.write("Not enough data for comparison.")
