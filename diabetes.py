import streamlit as st
import pandas as pd
from joblib import load
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix 

# Load the dataset
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split data for metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Load models
model_paths = {
    'XGBoost': 'models/xgboost.pkl',
    'Decision Tree': 'models/Decision_tree.pkl',
    'Random Forest': 'models/Random_Forest.pkl'
}
models = {name: load(path) for name, path in model_paths.items()}

# Calculate metrics
model_metrics = {
    name: {
        "accuracy": accuracy_score(y_test, model.predict(X_test)),
        "confusion_matrix": confusion_matrix(y_test, model.predict(X_test)).tolist()
    }
    for name, model in models.items()
}

# UI
st.title("🧠 Diabetes Prediction App with ML Models Test 2, Date 18/07/2025 .")



st.sidebar.header("Input Features")
chai=st.selectbox("Your fav chai:",["masala chai","Adhrak chai","Lemon Chai","Kesar Chai"])
st.write(f"Your choose{chai}.Excellent Choice")
st.success("Your Chai Has been Brewed.")

st.title("Chay Maker App")
st.button("Make Chai")
st.success("Chai is being made...")
add_masala=st.checkbox("I masala", value=True)
if add_masala:
    st.write("Masala has been added to your chai.")
    tea_type=st.radio("Select tea type:", ["Black Tea", "Green Tea", "Herbal Tea"])
    st.write(f"You have selected {tea_type}.")
    flavour=st.selectbox("choose a flavour:", ["Adhrakh", "Tulsi", "Kesar"])
    st.write(f"You have selected {flavour} flavour.")

sugar= st.slider("Sugar Level(Spoon)", 0, 10, 5)
st.write(f"Sugar level set to {sugar}.")  
Cups=st.number_input("Enter the number of cups:", min_value=1, max_value=10, value=1)    
st.write(f"You have selected {Cups} cup(s).")
name=st.text_input("Enter Your Name:")
if name:
    st.write(f"Hello,{name  }! Your chai is being prepared.")

dob=st.date_input("Enter Your Date of Birth is: ")
st.write(f"Your date of birth is {dob}")






    
model_choice = st.sidebar.selectbox("Choose a Model", list(model_paths.keys()))
Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0)
Glucose = st.sidebar.number_input("Glucose", min_value=0)
BloodPressure = st.sidebar.number_input("BloodPressure", min_value=0)
SkinThickness = st.sidebar.number_input("SkinThickness", min_value=0)
Insulin = st.sidebar.number_input("Insulin", min_value=0)
BMI = st.sidebar.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.sidebar.number_input("DiabetesPedigreeFunction", min_value=0.0)
Age = st.sidebar.number_input("Age", min_value=0)

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    })

    # Load and predict
    model = models[model_choice]
    prediction = model.predict(input_df)[0]
    result = "🩸 Diabetic" if prediction == 1 else "✅ Non-Diabetic"

    st.subheader("🔍 Prediction Result")
    st.write(result)

    # Show metrics
    acc = model_metrics[model_choice]["accuracy"]
    cm = model_metrics[model_choice]["confusion_matrix"]
    st.subheader("📊 Model Metrics")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.write("**Confusion Matrix:**")
    st.table(pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))
    # Log predictions
    input_df["Model"] = model_choice
    input_df["Prediction"] = result
    try:
        log_exists = pd.read_csv("prediction_logs.csv")
        input_df.to_csv("prediction_logs.csv", mode='a', header=False, index=False)
    except FileNotFoundError:
        input_df.to_csv("prediction_logs.csv", index=False)

    st.success("✅ Prediction logged successfully!")




    

