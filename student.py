import streamlit as st
import joblib
import numpy as np
import pandas as pd

#load the model
with open('models\student_performance_prediction.pkl','rb') as file:
    model = joblib.load(file)
    #title
st.set_page_config(page_title = " 🧑‍🎓Student Performance Prediction🧑‍🎓")
st.title(" 🈸Student  Performance Predictor App")
st.subheader("Predict the  Performance  of Student based on its Features")

#sidebar inputs
st.sidebar.header("Enter details of Students 😎")
#Area as slider
Hours =st.sidebar.slider("Hours Studied:",min_value=1 ,max_value=12,step=3)
#numerical Inputs
PreScores=st.sidebar.number_input("Previous Scores:",min_value=1,max_value=100,value=60)
ExtracurricularActivities=st.sidebar.radio("ExtracurricularActivities:", ["yes","no"] ,horizontal=True)
SleepHours=st.sidebar.number_input("Sleep Hours:",min_value=1,max_value=10,value=3)

Sampleqp=st.sidebar.number_input("Sample Question Papers Practiced:",min_value=1,max_value=5,value=2)



 # Prepare input
columns = ['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced' ]
#convert
ExtracurricularActivities=1 if ExtracurricularActivities.lower()=='yes' else 0

input_data = [[Hours,PreScores,ExtracurricularActivities,SleepHours,Sampleqp,]]

features_df = pd.DataFrame(input_data, columns=columns)

#button to prediction
if st.sidebar.button("Predict Student Performance"):
    try:
        prediction=model.predict(features_df)
        #display result
        st.success(f"Estimated Performance Predicted:{prediction[0]:.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed:{e}")
    #predict Performance
    #additional Info
    st.info("This prediction is based on a Multiple Linear Regression")

    #footer
    st.markdown('------')
    st.markdown("Made with using Streamlit🤩")

                                         

