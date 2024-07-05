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
from p import applyimputation,testcoorelationship,getAllViolinPlots,remove_singlevariate_outliers,oneHotEncoding
from p import info,getallcatfig,describe_column,getallconfigs,gettopnfeatures,violin_plot,oversampling,getClassificationReport



def main():
    # initial config
    imputeddf = pd.DataFrame()
    nooutlierdf = pd.DataFrame()
    onehotencodeddf = pd.DataFrame()
    sampleddf = pd.DataFrame()
    scaleddf=pd.DataFrame()


    kdf = st.session_state.df
    target = st.session_state.target
    # kdf = pd.read_csv("data.csv")
    # target = "Diagnosis"

    datetime_columns = kdf.select_dtypes(include=[pd.DatetimeTZDtype, 'datetime']).columns

    # Remove datetime columns
    kdf = kdf.drop(columns=datetime_columns)
    
    
    # Remove columns with all NaN values
    kdf = kdf.dropna(axis=1, how='all') 
     # imputing all missing values in every columns
    
    cat_cols = [col for col in kdf.columns if kdf[col].nunique() < 10]
    num_cols = [col for col in kdf.columns if col not in cat_cols]
    cat_cols.remove(target)
    totalcols = kdf.columns.tolist()
    noofcols = len(totalcols)
    
    # RENMOVING UNNECESSARY STRING TYPE COLUMN FROM NUMERICAL COLUMNS
    string_columns = kdf.select_dtypes(include=['object']).columns.tolist()
    newlist = []

    for col in  string_columns:
        if col in num_cols:
            newlist.append(col)
        
    kdf.drop(columns=newlist,axis=1,inplace=True)



    st.title("pre processing page",anchor=False)
    quote = "Without data you are another person with an opnion"
    author = "W.Edward's Duming"

    # Display the quote using Markdown
    st.markdown(f"""
        <div style="background-color:#0E1117;padding:20px;border-radius:10px;">
            <p style="font-size:24px;font-style:italic;text-align:center;">"{quote}"</p>
            <p style="font-size:20px;text-align:right;margin-right:20px;">- {author}</p>
        </div>
    """, unsafe_allow_html=True)

    if target in totalcols:
        st.text("Scroll to bottom to download the pre-processed dataset")
        
        
        
        # imputation of dataset
        with st.container(border=True):
            st.write(f"1st basic step involves understanding the features of data and to impute any NULL values. Here to impute any categorical column we have used *simple imputer with most frequent strategy* and impute numerical data we have used *simple imputer with median strategy*")
            st.write(np.round(kdf.describe(),1),height=400)
    
            imputeddf,oldreport,newreport = applyimputation(df=kdf)
            st.text(" ")
            st.text("Before imputation percentage of null values")
            st.dataframe(oldreport)
            st.text("After Imputation percentage of null values")
            st.dataframe(newreport)
            
                # assigning cleaned dataset to session state for further use
            st.session_state.df = kdf
            
            # GETTING THE TOP N FEATURES IN SESSION STATE
            most = testcoorelationship(imputeddf,totalcols,target)
            most.remove(target)
            st.session_state.topnfeatures =  most
            
            
            # using iqr method to remove outlier
        st.subheader("2nd Step : To remove both single variate and multi-variate outliers from the data-frame")

        with st.container(border=True):
            st.subheader("Outlier Detection",anchor=False)
            
            st.write("Using IQR  method to remove single variate outlier")
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*ZrRgmtVHMVLknr7BmezXlg.jpeg")
            st.write(' ')
            hui = totalcols
            hui.remove(target)
            print("after outlier ", imputeddf.shape)
            # print("hui",type(totalcols))
            nooutlierdf = remove_singlevariate_outliers(df=imputeddf,columns=hui)
            print("after outlier ", nooutlierdf.shape)
            # print(nooutlierdf.shape)
            # st.write("*********Using Mahalanobis  distance to remove multivariate outliers**********")
            # st.write("Mahalanobis distance is a statistical technique used to identify and remove outliers in multivariate data (data with multiple variables)")
            # st.image("https://miro.medium.com/v2/resize:fit:1400/1*Zj_jFn6SfDPwmasUBCAR1A.png")
                
        st.write("3rd Step : ONE_HOT_ENCODING")       
        with st.container(border=True):
            st.write(f"Machine learning algorithms generally work better with numerical input rather than categorical data. One-hot encoding converts categorical variables into a format that can be used by these algorithms.By dropping the first category, we prevent multicollinearity in our model.Using dtype=np.int32 reduces the memory usage")
            onehotencodeddf = oneHotEncoding(df=nooutlierdf,numcols=num_cols,catcols=cat_cols)
            st.dataframe(onehotencodeddf.head())       
                
                
        # with st.container(border=True):
        #     st.subheader("Data Binning",anchor=False)
            
        #     st.write("*Data binning, also known as data discretization, is a technique used in data preprocessing that groups continuous values into discrete intervals or bins. This process helps in reducing the effects of minor observation errors and enhances the performance of models by simplifying the data.*")    
            
        #     st.write("__Here Age column has been binned into three bins Teenage , Adult and Senior__ ")
            
            # over or under sampling dataset
        with st.container(border=True):
            st.subheader("SMOTE",anchor=False)
            
            info = onehotencodeddf[target].value_counts()
            st.write("*SMOTE is an oversampling technique that creates synthetic samples of the minority class by interpolating between existing minority class instances.The goal of SMOTE is to balance the class distribution by generating synthetic examples rather than by duplicating existing ones, which helps prevent overfitting.*")     
            st.write("__Before oversampling__")
            
            st.dataframe(info) 
            sampleddf = oversampling(df=onehotencodeddf,target=target)
            st.write("__After Oversampling__")
            # print(sampleddf.info())
            newinfo = sampleddf[target].value_counts()
            st.dataframe(newinfo) 
            
            # The encoded dataset is now scaled and fitted using random forestclassifier 
            
            
        with st.container(border=True):
            st.subheader("Min-max scaling",anchor=False)
            st.write("*Min-max scaling, also known as min-max normalization, transforms numerical features to a common scale, typically [0, 1]. It rescales features by shifting and scaling values to a specified range, where the minimum value of the feature becomes 0 and the maximum value becomes . Min-max scaling preserves the original distribution and relationships between data points.*") 
            st.text(" ")
            st.write("__Classification Report after Training a Random Forest Model__")   
            
            report,scaleddf = getClassificationReport(df=sampleddf,target=target)
            st.dataframe(scaleddf.head())    
            csv_file = StringIO(report)
            # Convert the string to a DataFrame
            reportdf = pd.read_csv(csv_file)
            st.dataframe(reportdf)
            

        st.text( "  ")
        if  not scaleddf.empty:
            filecontent = scaleddf.to_csv()
            st.download_button(label='Download the Pre-Processed Dataset',
                            data = filecontent,
                            type='primary',
                            file_name=f"preprocessed{target}.csv")
            
    else :
        st.warning("Enter correct Target column name (it is case sensitive)")        
        
        
        
        
        
if __name__ == "__main__":
    main()           
        
        
        