import pandas as pd
import streamlit as st
import plotly.express as px # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import textwrap
import requests
import asyncio
import json
import aiohttp



# aws lambda endpoint
api_url = "https://oiffrmbzm4.execute-api.ap-south-1.amazonaws.com/dev/intel"  


# there may be some cases some noise like these error string come with our result we need to remove it
errorstring1 = " Agent stopped due to iteration limit or time limit."
errorstring2  = "Missing: everything | Show results with"
errorstring3 = "..."

# function to display container with result
def display_batches(batches):
    linewidth = 20
    for batch in batches:
        col = batch['col']
        res = batch['response']
        res = res.replace(errorstring1," ")
        res = res.replace(errorstring2," ")
        res = res.replace(errorstring3," ")
        
        wrapped_text = textwrap.fill(res, width=linewidth)
        with st.container(border=True):
            st.subheader(f"Insights for the column {col}")
            st.text(" ")
            st.write(wrapped_text)
            
# function to display container with cached result
            
def displaycachedresult(batches):
    linewidth = 20
    for batch in batches:
        col = batch['col']
        res = batch['response']
        res = res.replace(errorstring1," ")
        res = res.replace(errorstring2," ")
        res = res.replace(errorstring3," ")
        
        wrapped_text = textwrap.fill(res, width=linewidth)
        with st.container(border=True):
            st.subheader(f"Insights for the column {col}")
            st.text(" ")
            st.write(wrapped_text)  
            
            
                      
# FUNCTION TO MAKE POST REQUEST TO API URL ENDPOINT
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


# batching queries to be sent to aws lambda

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
               
    st.session_state.isDone = True     
    st.session_state.stored_result = results   
    print(results)


async def main():
    st.title("Insights 💡📊 with the help of AI Agents leveraged with the help of AWS S3 and AWS Lambda Service")
    if st.session_state.df is not None:
        
        # checking if all insights are gathered or not
        if "isDone" not in st.session_state:
            st.session_state.isDone = False
            
            
        #checking if session has all the stored result 
        if "stored_result" not in st.session_state:
            st.session_state.stored_result =[]
            
                
        if(not st.session_state.isDone):
            st.text("Please Wait.... It may take a few minutes to generate insights")
        kdf= st.session_state.df
        target = st.session_state.target
        
        filename = st.session_state.filename
        datasetinfo = st.session_state.datasetinfo
        
        # filename = "tejaswee"
        column_dict = { }
        # datasetinfo = overview
        # kdf = pd.read_csv("data.csv")
        # target = "Diagnosis"

        cat_cols = [col for col in kdf.columns if kdf[col].nunique() < 10]
        num_cols = [col for col in kdf.columns if col not in cat_cols]
        cat_cols.remove(target)
        totalcols = kdf.columns.tolist()
        noofcols = len(totalcols)
        
        # top n related column to generate insights for
        insight_cols = st.session_state.topnfeatures
        # using session variables to cache the result
        if st.session_state.isDone == False:
            try:
                await batching(insight_cols,column_dict,filename,datasetinfo,target)
            except Exception as e :
                st.text(f"error in main of insights.py file{e} ")
        else:
            displaycachedresult(st.session_state.stored_result)        
    else:
        st.warning("PLEASE UPLOAD YOUR DATASET FIRST")        
        
if __name__ == "__main__":
    asyncio.run(main())          