from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonREPLTool
import pandas as pd
import os
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
import streamlit as st
import os
from PIL import Image
import streamlit as st
import random
import string
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
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



def helpcsv (question,dffilename,imagename):
    df = pd.read_csv(dffilename)
    pythontools = [PythonREPLTool()]
    conversation_with_summary = ConversationChain(
        llm=llm,    
        memory=ConversationBufferWindowMemory(k=2),
        verbose=True
    )
    filename  = "csv/test.csv"

    prompt_visualize_csv = PromptTemplate.from_template(
        "you are skillfull csv reader using pandas and pythons tools. GENERATE the necessary python code for the query {question} donot write the string python at the starting assuming name of file is {name} and to save the figure of plot  use the filename as {image} and also give the output in python multiline string format. donot run more than three times  please donot give any error ."
    )
    # toquery =f'GENERATE the python code for the query {question} assuming name of file is {name} and to save the figure of plot  use the filename as {image}.'


    
    agent_pandas = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,

    )
    
    
    imagepath = f"csv/{imagename}"
    name = "csv/test.csv"
    toquery =f'GENERATE the python code for the query {question} assuming name of file is {name} and to save the figure of plot  use the filename as {imagepath}.'
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





def chatcsv(question,filepath):
    # openaikey = os.environ.get("OPENAI_API_KEY")
    filename = "kidney.csv"
    pythontools = [PythonREPLTool()]
    conversation_with_summary = ConversationChain(
        llm=llm,    
        memory=ConversationBufferWindowMemory(k=2),
        verbose=True
    )
    query_prompt_template = PromptTemplate.from_template(
    """you are skillfull csv reader using pandas and pythons tools. So answer the question {question} based on the csv file given.name of the csv file is {filepath}
    if you need any further information to answer the question please ask 
    """)
    df = pd.read_csv(filepath)
    agent_pandas = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        tool = [pythontools],
        verbose=True,
    )
    agent = query_prompt_template  | agent_pandas 
    response = agent.invoke({"question":question,"filepath":filepath})
    print (response["output"])
    res = response["output"]
    print(res)
    return  res




def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'query'
    def navigate(page):
        st.session_state.page = page    
        
    st.set_page_config(page_title="Ask your CSV")
        
    st.sidebar.title("Query CSV")
    st.sidebar.button("Chat With CSV", on_click=navigate, args=("query",))
    st.sidebar.button("Visualize Your CSV", on_click=navigate, args=("visualize",))    
        
    if st.session_state.page == 'query':
        # st.session_state.df = None
        st.title("Home")
        st.write("Welcome to the Home page.")
        st.header("Ask your CSV 📈")
        # if st.session_state.df is None : 
        #     csv_file = st.file_uploader("Upload a CSV file", type="csv")
            
        #     if csv_file is not None:
        #         df = pd.read_csv(csv_file)
        #         st.write("Uploaded CSV file:")
                

        #         with open("csv/test.csv", "wb") as f:
        #             f.write(csv_file.getbuffer())
                
        #         st.success("File saved successfully as 'uploaded_file.csv'")
        #         st.session_state.df = df
        #     else:
        #         st.write("No file uploaded yet.")
        
        
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What is up?"):
            st.chat_message("user").markdown(prompt)
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            response = chatcsv(prompt,"csv/test.csv")

            response = f"AI: {response}"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        
    elif st.session_state.page == 'visualize':
        st.warning("This Feature of app is a Beta Version So some error might come please do reload the site")
        if 'images' not in st.session_state:
            st.session_state.images = []
        # df = st.session_state.df
        st.title("Page 1")
  
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
            dffilename  = "csv/test.csv"
            check = helpcsv(question=prompt,dffilename=dffilename,imagename=filename)
            image = None
            res = "csv/8aAR.png"
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

if __name__ == "__main__":
    main()