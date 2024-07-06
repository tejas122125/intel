from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonREPLTool
import pandas as pd
import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
import streamlit as st
from PIL import Image
import random
import string
from dotenv import load_dotenv
 
# Load the environment variables from the .env file FOR THE GOOGLE AI LLM
load_dotenv()
google = os.getenv('GOOGLEAI')
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google,temperature=0.4)


def stringrun (code):
    try:
        code = code
        exec(code)
    except:
        print("execptiopn occured no worry")    


def generate_random_string(length=4):
    # Generate a random string of the specified length
    letters_and_digits = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(letters_and_digits) for i in range(length))
    return random_string


# FUNCTION TO HELP VISUALIZE CSV
def helpcsv (question,imagename):
    df = st.session_state.df
    pythontools = [PythonREPLTool()]

    # prompt_visualize_csv = PromptTemplate.from_template(
    #     "you are skillfull csv reader using pandas and pythons tools. GENERATE the necessary python code for the query {question} donot write the string python at the starting assuming name of file is {name} and to save the figure of plot  use the filename as {image} and also give the output in python multiline string format. donot run more than three times  please donot give any error ."
    # )
    # toquery =f'GENERATE the python code for the query {question} assuming name of file is {name} and to save the figure of plot  use the filename as {image}.'


    # INITIALIZING AGENT PANDAS FROM LANGCHAIN TO GENERATE PLOTS
    agent_pandas = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        allow_dangerous_code=True
    )
    
    imagepath = f"csv/{imagename}"
    
    toquery =f'GENERATE the python code for the query {question} to save the figure of plot  use the filename as {imagepath}.'
    try:
        # agent = prompt_visualize_csv  | agent_pandas 
        # response = agent.invoke({"question":question,"name":"test.csv","image":imagepath})
        response = agent_pandas.invoke(toquery)
        # print (response["output"])
        res = response["output"]
        res =  res.replace("python", "")
        print(res)
        stringrun(res)
        return True
    except Exception as e:
        return False    


# FUNCTION TO HELP QUERY YOUR CSV
def chatcsv(question):
    pythontools = [PythonREPLTool()]
    
    # query_prompt_template = PromptTemplate.from_template(
    # """you are skillfull csv reader using pandas and pythons tools. So answer the question {question} based on the csv file given.name of the csv file is {filepath}
    # if you need any further information to answer the question please ask 
    # """)
    
    
    # INITIALIZING PANDAS AGENT FROM LANGCHAIN 
    df = st.session_state.df
    print(df.head())
    
    agent_pandas = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        tool = [pythontools],
        verbose=True,
        allow_dangerous_code=True
    )
    response = agent_pandas.invoke(
    f"""you are skillfull csv reader using pandas and pythons tools. So answer the question {question} based on the csv file given
    if you need any further information to answer the question please ask 
    """)

    print (response["output"])
    res = response["output"]
    print(res)
    return  res




def main():
    
    
    st.title("Ask Your Dataset 📈",anchor=False)        
    st.sidebar.title("Query Dataset")
    
    if st.session_state.df is not None:
        # check for query or visualize page
        if 'page' not in st.session_state:
            st.session_state.page = 'query'
        def navigate(page):
            st.session_state.page = page    
            

        st.sidebar.button("Chat With Your Dataset", on_click=navigate, args=("query",))
        st.sidebar.button("Visualize Your Dataset", on_click=navigate, args=("visualize",))    
            
            
            # PAGE FOR QUESTIONING YOUR DATASET
        if st.session_state.page == 'query':
            # st.session_state.df = None
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # React to user input
            if prompt := st.chat_input("What is up?"):
                st.chat_message("user").markdown(prompt)
                
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                response = chatcsv(prompt)

                response = f"AI: {response}"
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            # PAGE FOR VISUALIZING YOUR DATASETS (BETA VERSION ERROR MAY OCCUR)
        elif st.session_state.page == 'visualize':
            st.warning("This Feature of app is a Beta Version So some error might come please do reload the site")
            if 'images' not in st.session_state:
                st.session_state.images = []
            # df = st.session_state.df
    
            if "visualmessages" not in st.session_state:
                st.session_state.visualmessages = []

            for message in st.session_state.visualmessages:
                with st.chat_message(message["role"]):
                    if message['role'] == "assistant":
                        fname = message['content']
                        img = Image.open(fname,mode="r")
                        st.image(img)
                    else:    
                        st.markdown(message["content"])
                    

    #         # React to user input
            if prompt := st.chat_input("What is up?"):
                st.chat_message("user").markdown(prompt)
                st.session_state.visualmessages.append({"role": "user", "content": prompt})
                
                random_str = generate_random_string()
                filename = f"{random_str}.png"
                check = helpcsv(question=prompt,imagename=filename)
                image = None
                res = "csv/fun.png"
                if check :
                    try :
                        image = Image.open(f'csv/{filename}')
                        st.session_state.images.append(image)
                        
                    except :
                        response = "  "    
        
                # Store the image in session state
                    res = f'csv/{filename}'
                    response = "here is your plot"
                else :
                    response = "some error occured"

                # response = f"AI: {response}"
                with st.chat_message("assistant"):
                    if check:
                        st.image(image)
                        st.markdown(response)
                    else:
                        st.markdown(response)    
                st.session_state.visualmessages.append({"role": "assistant", "content": res})           
    else:
        st.warning("PLEASE UPLOAD YOUR DATASET FIRST")            



if __name__ == "__main__":
    main()