import boto3
import pandas as pd
import json
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonREPLTool
import pandas as pd
import os
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import AgentType

from langchain_google_genai import GoogleGenerativeAI

import os
import pandas as pd
import numpy as np


# LIST OF GOOGLE API KEY FOR GENERATIVE AI LLM
keys= []

# INITIALIZING THE GOOGLE LLM MODELS
def getllms():
    all_llm = []
    lenkey = len(keys)
    for i in range(lenkey):
        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key = keys[i],temperature=0.9)
        all_llm.append(llm)
    return all_llm    

# USED TO DOWNLOAD FROM AWS S3
def download_from_s3(bucket_name ="yt-lambda-layer", object_name = "tejaswee.csv"):
    
    AWSACCESSKEYID = "YOUR_AWS_ACCESS_ID"
    AWSSECRETKEYID  = "YOUR_SECRET_ID"

    s3_client = boto3.client(
    's3',
    aws_access_key_id=AWSACCESSKEYID,
    aws_secret_access_key=AWSSECRETKEYID,
    region_name='ap-south-1'
 )
    try:
        # Download the file
        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
        file_content = response['Body'].read()
        # STORING THE DOWNLOADED FILE IN THE TMP FOLDER OF THE DOCKER CONTAINER
        with open("/tmp/tej.csv","wb") as test:
            test.write(file_content)
        # st.dataframe(df)
        df = pd.read_csv("/tmp/tej.csv")
        # print(df.head())
        return df
        
 
    except Exception as e:
        print(e)
        return None


# HANDLES THE API CALL TO GOOGLE LLM MODEL AND GIVES THE RESPONSE BACK TO THE STREAMLIT UI WITH ERROR HANDLING
def handleverything(idxllm1,idxllm2,dfname,dfinfo,col,colinfo,target):
    all_llm  = getllms()
    llm1 = all_llm[idxllm1]
    llm2 = all_llm[idxllm2]
    dfpath = f'{dfname}.csv'
    df = download_from_s3(bucket_name="yt-lambda-layer",object_name=dfpath)
    os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY"
    search = GoogleSerperAPIWrapper()
    query =f'everything about {col}'
    answer =" "
    ans = " "
    # serper = load_tools(["google-serper"], llm=llm2)
    agent_pandas = create_pandas_dataframe_agent(
            llm=llm1,
            max_iterations=10,
            df=df,
            verbose=True,
            agent_executor_kwargs={"handle_parsing_errors": True,"extra_tool" : [PythonREPLTool]},
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code = True
            )
    try:
        result = agent_pandas.invoke(f""" PLEASE generate output having more than 100 words . give me some detailed general trends or insights about the column {col} like what do they represent in real world with respect to {target} column in the datframe. You can use information of dataset {dfinfo} and column information {colinfo}""")
        
        answer = result['output']
        
    except Exception as e:
        print(e)
        err = str(e)
        words = err.split()
        remaining_words = words[20:]

        result = ' '.join(remaining_words)
        answer = result
             
    try:           
        ans =search.run(query)
        # agentsearch = initialize_agent(
        #         serper, llm2, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,agent_executor_kwargs={"handle_parsing_errors": True}
        #         ) 
        # res = agentsearch.invoke(f""" using the dataset information{dfinfo} and column information {colinfo} help me  by generating only two  good quality  questions.After then use the tool to generate answer and IMPORTANT return me a good summarized answer. do it really fast""")
        # ans = res['output']
    except Exception as e:
        print("error in handle everything function while searching try",e)
        ans = " "
            
    
    finalanswer = ans + " " + answer
    return finalanswer


# STANDARD FUNCTION IN AWS LAMBDA TO RECIEVE A REQUEST FROM STREAMLIT UI
def handler(event,context):
    print(event)
    bui = event
    print(bui)
    idxllm1 = bui['idxllm1']
    idxllm2 = bui['idxllm2']
    dfname = bui['dfname']
    dfinfo = bui['dfinfo']
    col=bui['col']
    colinfo = bui['colinfo']
    target = bui['target']
    res =" "
    
    try:
        res = handleverything(idxllm1=idxllm1,idxllm2=idxllm2,dfname=dfname,dfinfo=dfinfo,col=col,colinfo=colinfo,target=target)
        print("finalresultttttttt isd ",res)


    except Exception as e :
        err = str(e)
        res = err

    return {"statusCode":200,"body":{
        "response":res
    }}     
    # print("length of records poolled",len(event['Records']))
    # print("here ia m testing ",event['Records'][0]['body'])
    # idxllm1,idxllm2,dfname,dfinfo,col,colinfo,target
