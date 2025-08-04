import streamlit as st
import joblib
import numpy as np
import pandas as pd

#load the model
with open('models\houseprice.pkl','rb') as file:
    model = joblib.load(file)

#title
st.set_page_config(page_title = " House Price Prediction")
st.title("🏡 House Price Predictor App")
st.subheader("Predict the  Price  of house based on its Features")

#sidebar inputs
st.sidebar.header("Enter House details😎")

#Area as slider
sqft_living=st.sidebar.slider("sqft_living(in square feet):",min_value=10 ,max_value=100000,step=10)
#numerical Inputs
bathrooms=st.sidebar.number_input("Number of Bathrooms:",min_value=1,max_value=10,value=3)
bedrooms=st.sidebar.number_input("Number of Bedrooms:",min_value=1,max_value=10,value=3)
grade=st.sidebar.number_input("grade:",min_value=1,max_value=10,value=3)
id=st.sidebar.text_input("Id:", value=100000010)
date=st.sidebar.text_input("Date:",value="20140502T000000")
price=st.sidebar.number_input("price:",min_value=1,max_value=100000,value=3)
sqft_lot=st.sidebar.slider("sqft_lot(in square feet):",min_value=10 ,max_value=10000,step=10)
floors=st.sidebar.number_input("Number of Floors:",min_value=1,max_value=10,value=3)
waterfront=st.sidebar.number_input("Waterfront:",min_value=0,max_value=0,value=0)
view=st.sidebar.number_input("View:",min_value=0,max_value=0,value=0)
sqft_basement=st.sidebar.slider("sqft_basement(in square feet):",min_value=0 ,max_value=1000,step=10)
yr_built=st.sidebar.number_input("Year Built:",min_value=0,max_value=20000,value=1)
yr_renovated=st.sidebar.number_input("renovated year:",min_value=0,max_value=2000,value=5)
zipcode=st.sidebar.text_input("Number of zipcode:",value="98178")
lat=st.sidebar.number_input("lat:",min_value=1,max_value=10,value=3)
long=st.sidebar.number_input("long:",min_value=1,max_value=10,value=3)
#Area as slider
sqft_above=st.sidebar.slider("sqft_above(in square feet):",min_value=10 ,max_value=100000,step=10)
sqft_living15=st.sidebar.slider("sqft_living15(in square feet):",min_value=10 ,max_value=100000,step=10)
sqft_lot15=st.sidebar.slider("sqft_lot15(in square feet):",min_value=10 ,max_value=100000,step=10)
condition = st.sidebar.number_input("Condition:", min_value=0, max_value=0, value=0)

 
 # Prepare input
columns = ['sqft_living','bathrooms','bedrooms','grade','id','date','price', 'sqft_lot','floors','waterfront','view',
           'sqft_basement','yr_built','yr_renovated' ,'zipcode','lat','long' , 'sqft_above','sqft_living15','sqft_lot15','condition' ]

input_data = [[sqft_living,bathrooms,bedrooms,grade,id,date,price, sqft_lot,floors,waterfront,view,
           sqft_basement,yr_built,yr_renovated ,zipcode,lat,long , sqft_above,sqft_living15,sqft_lot15,condition]]

features_df = pd.DataFrame(input_data, columns=columns)


#button to prediction
if st.sidebar.button("Predict House  Price"):
    try:
        prediction=model.predict(features_df)
        #display result
        st.success(f"Estimated House Price: Rs{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed:{e}")
    #predict Price
    #additional Info
    st.info("This prediction is based on a Multiple Linear Regression")

    #footer
    st.markdown('------')
    st.markdown("Made with using Streamlit🤩")