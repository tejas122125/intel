import streamlit as st
import graphviz as gv
from io import StringIO
import pandas as pd
import streamlit as st
import plotly.express as px # type: ignore
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.optimize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import requests
import asyncio
import json
import aiohttp
# aws lambda endpoint
api_url = "https://oiffrmbzm4.execute-api.ap-south-1.amazonaws.com/dev/intel"  


overview  ="overview:This dataset contains detailed health information for 1,659 patients diagnosed with Chronic Kidney Disease (CKD).."



def display_batches(batches):
    for batch in batches:
        col = batch['col']
        res = batch['response']
        with st.container(border=True):
            st.subheader(f"Insights for the column {col}")
            st.text(" ")
            st.text(res)
            


async def make_post_request(idxllm1=0,idxllm2=0,dfname="tejaswee",dfinfo=" ",col=" ",colinfo=" ",target= " "):
    data = {
    "idxllm1":idxllm1,
    "idxllm2":idxllm2,
    "dfname":dfname,
    "dfinfo":dfinfo,
    "col":col,
    "colinfo":colinfo,
    "target":target
}
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, json=data) as response:
            result = await response.json()
            return {"col":col,"response":result['body']['response']}




async def batching(cols =[],cols_info ={},dfname="tejaswee",dfinfo=" ",target= " "):
    count = 0
    batch_size =5
   
    countsize = 10
    results =[]
    

    for i in range(0, len(cols), batch_size):
        batch = cols[i:i + batch_size]
        # print(f"Batch {i // batch_size + 1}:")
        task =[]
        for col in batch:
            columninfo =" "
            # try:
            #     columninfo = cols_info[col]
            # except Exception as e:
            #     print(e)
            #     columninfo = " " 
                   
            task.append(make_post_request(count,count,dfname,dfinfo,col,columninfo,target))
            print(col,count)
            count = (count+1) % countsize
        try:    
            result = await asyncio.gather(*task) 
            display_batches(result)
            print(result)
            
            results = results + result   
        except Exception as e :
            print("error while gathering",e)
               
            
    print(results)


async def main():
    st.header("Insight with the help of AI Agents leveraged with the help of AWS S3 and AWS Lambda Service")
    st.text("Please Wait.... It may take a few minutes to generate insights")
    # df= st.session_state.df
    # target = st.session_state.target
    
    # filename = st.session_state.filename
    # datasetinfo = st.session_state.datasetinfo
    
    filename = "tejaswee"
    column_dict = { }
    datasetinfo = overview
    kdf = pd.read_csv("data.csv")
    target = "Diagnosis"

    cat_cols = [col for col in kdf.columns if kdf[col].nunique() < 10]
    num_cols = [col for col in kdf.columns if col not in cat_cols]
    cat_cols.remove(target)
    totalcols = kdf.columns.tolist()
    noofcols = len(totalcols)
    
    # top n related column to generate insights for
    insight_cols = st.session_state.topnfeatures
    
    try:
        await batching(insight_cols,column_dict,filename,datasetinfo,target)
    except Exception as e :
        st.text("error in main of insights.py file ", str(e))
        
        
if __name__ == "__main__":
    asyncio.run(main())          