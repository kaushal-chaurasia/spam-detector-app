import streamlit as st
import pandas as pd
st.title("Chai Dashboard")
file=st.file_uploader("Upload your data file", type=["csv", "xlsx"])
if file:
    df= pd.read_csv(file)
    st.subheader("Data Preview")
    st.dataframe(df)
    st.success("File uploaded successfully!")
if file:
    st.subheader("Data Statistics")
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])
    st.write("Columns:", df.columns.tolist())
    st.write("Data Types:")
    st.dataframe(df.dtypes)
    st.write("Data Description:")
    st.dataframe(df.describe())