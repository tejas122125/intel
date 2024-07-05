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



# LOADING AWS KEYS AND INITIALIZING BOTO3 TO CONNECT WITH AWS
load_dotenv()
awsaccesskeyid = os.getenv('AWSACCESSKEYID')
awssecretkeyid = os.getenv('AWSSECRETKEYID')

s3_client = boto3.client(
    's3',
    aws_access_key_id=awsaccesskeyid,
    aws_secret_access_key=awssecretkeyid,
    region_name='ap-south-1'
)

# # 
# def download_from_s3(bucket_name, object_name):
#     try:
#         # Download the file
#         response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
#         file_content = response['Body'].read()
#         # df = pd.read_csv(file_content)
#         st.text("downloading the dataframe")
#         with open("tej.csv","wb") as test:
#             test.write(file_content)
#         # st.dataframe(df)
#         return file_content
#     except Exception as e:
#         st.error(f"Error: {e}")
#         return None
    
    
#UPLOADING TO AWS S3 THE GIVEN DATASET 
def upload_to_s3(file, bucket_name, object_name):
    object_name = f"{object_name}.csv"
    try:
        with open(file,"rb") as cs :
            s3_client.upload_fileobj(cs, bucket_name, object_name)
            st.success(f"Successfully uploaded {object_name} to AWS S3")
    except Exception as e:
        st.error(f"Error: {e}")

def main():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to InsightMaster👋")
    tempdf = pd.DataFrame()
    
    
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'filename' not in st.session_state:
        st.session_state.filename = ''
    if 'datasetinfo' not in st.session_state:
        st.session_state.datasetinfo = ''
    if 'target' not in st.session_state:
        st.session_state.target = ''



    st.session_state.filename = st.text_input("Enter a name of the dataset",value=st.session_state.filename)
    st.session_state.datasetinfo = st.text_input("Enter a description of the dataset",value=st.session_state.datasetinfo)

    csv_file  = None

    if st.session_state.df is None : 
        csv_file = st.file_uploader("Upload a CSV file", type="csv")
        if csv_file is not None:
            tempdf = pd.read_csv(csv_file)
            with open("csv/test.csv", "wb") as f:
                f.write(csv_file.getbuffer())
                
            # st.session_state.df = df1
            cols = tempdf.columns.to_list()
            target = st.selectbox("Select target column name",cols)
            st.session_state.target = target
        

        

    if st.button("SUBMIT"):
        if  st.session_state.filename  and st.session_state.datasetinfo  and st.session_state.target and csv_file:
            
            if csv_file is not None:
                st.session_state.df = tempdf
                # st.write("Uploaded CSV file:")
                # with open("csv/test.csv", "wb") as f:
                #     f.write(csv_file.getbuffer())
                    
                # st.session_state.df = df
                
                bucket_name = "yt-lambda-layer"
                object_name = st.session_state.filename
                
                upload_to_s3("csv/test.csv", bucket_name, object_name)
                
                st.info("after submiting the dataset please first go to preprocessing page to clean your dataset for further analysis")

            else:
                st.write("No file uploaded yet.")
        
if __name__ == "__main__":
    main()           

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