# tejaswee testing plots and data preprocessing
import streamlit as st
import plotly.express as px # type: ignore
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image

import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Streamlit 👋")

st.session_state.df = None
if st.session_state.df is None : 
    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.write("Uploaded CSV file:")
        

        with open("csv/test.csv", "wb") as f:
            f.write(csv_file.getbuffer())
        
        st.success("File saved successfully as 'uploaded_file.csv'")
        st.session_state.df = df
    else:
        st.write("No file uploaded yet.")

target = st.text_input("ENTER CORRECT NAME OF THE TARGET COLUMN FOR CLASSIFICATION")

if st.button('Submit'):
    st.session_state.target = target

st.markdown("[Go to Page 1](pre)")
image = Image.open("csv/AOkF.png")
st.image(image)

st.sidebar.success("Select a demo above.")