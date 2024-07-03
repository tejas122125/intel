# tejaswee testing plots and data preprocessing
import streamlit as st
import plotly.express as px # type: ignore
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import boto3
from io import StringIO
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
awsaccesskeyid = os.getenv('AWSACCESSKEYID')
awssecretkeyid = os.getenv('AWSSECRETKEYID')

s3_client = boto3.client(
    's3',
    aws_access_key_id=awsaccesskeyid,
    aws_secret_access_key=awssecretkeyid,
    region_name='ap-south-1'
)
def download_from_s3(bucket_name, object_name):
    try:
        # Download the file
        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
        file_content = response['Body'].read()
        # df = pd.read_csv(file_content)
        st.text("downloading the dataframe")
        with open("tej.csv","wb") as test:
            test.write(file_content)
        # st.dataframe(df)
        return file_content
    except Exception as e:
        st.error(f"Error: {e}")
        return None
    
def upload_to_s3(file, bucket_name, object_name):
    object_name = f"{object_name}.csv"
    try:
        with open(file,"rb") as cs :
            s3_client.upload_fileobj(cs, bucket_name, object_name)
            st.success(f"Successfully uploaded {object_name} to {bucket_name}")
    except Exception as e:
        st.error(f"Error: {e}")


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Streamlit 👋")

st.session_state.df = None
st.session_state.filename = None
st.session_state.columnsinfo = None
st.session_state.datasetinfo = None
st.session_state.target = None


filename = st.text_input("Enter a name of the dataset")

st.info("""PLEASE upload columns-description in this format :
        
Age: The age of the patients ranges from 20 to 90 years.\n
Gender: Gender of the patients, where 0 represents Male and 1 represents Female.\n
Ethnicity: The ethnicity of the patients, coded as follows:
0: Caucasian
1: African American
2: Asian
3: Other.  """)
columnsinfo = st.text_area("Enter description of each column")
datasetinfo = st.text_area("Enter a description of the dataset")
target = st.text_input("ENTER CORRECT NAME OF THE TARGET COLUMN FOR CLASSIFICATION") 

st.session_state.filename = filename
st.session_state.columnsinfo = columnsinfo
st.session_state.datasetinfo = datasetinfo

csv_file  = None

if st.session_state.df is None : 
    csv_file = st.file_uploader("Upload a CSV file", type="csv")

if st.button("SUBMIT"):
    if filename and columnsinfo and datasetinfo and csv_file:
        
        if csv_file is not None:
            
            df = pd.read_csv(csv_file)
            st.write("Uploaded CSV file:")
            with open("csv/test.csv", "wb") as f:
                f.write(csv_file.getbuffer())
                
            st.session_state.df = df
            
            bucket_name = "yt-lambda-layer"
            object_name = st.session_state.filename
            
            upload_to_s3("csv/test.csv", bucket_name, object_name)

        else:
            st.write("No file uploaded yet.")


# st.markdown("[Go to Page 1](pre)")
# image = Image.open("csv/AOkF.png")
# st.image(image)

# st.sidebar.success("Select a demo above.")

# bucket_name = st.text_input("Enter the S3 bucket name:")
# object_name = st.text_input("Enter the file name in S3:")

# if st.button("Download"):
#     if bucket_name and object_name:
#         file_content = download_from_s3(bucket_name, object_name)
#         if file_content:
#             st.success(f"Successfully downloaded {object_name} from {bucket_name}")
#             st.download_button(
#                 label="Download File",
#                 data=file_content,
#                 file_name=object_name
#             )
#     else:
#         st.warning("Please provide both bucket name and file name.")