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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error



def main():

    
    st.title("Pre-Processing Page 🧮",anchor=False)
    
    
    if st.session_state.df is not None :
        
        
        quote = "Without data you are another person with an opnion"
        author = "W.Edward's Duming"

        # Display the quote using Markdown
        st.markdown(f"""
            <div style="background-color:#0E1117;padding:20px;border-radius:10px;">
                <p style="font-size:24px;font-style:italic;text-align:center;">"{quote}"</p>
                <p style="font-size:20px;text-align:right;margin-right:20px;">- {author}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # initial config
        imputeddf = pd.DataFrame()
        nooutlierdf = pd.DataFrame()
        onehotencodeddf = pd.DataFrame()
        sampleddf = pd.DataFrame()
        scaleddf=pd.DataFrame()
        edadf = pd.DataFrame()


        kdf = st.session_state.df.copy()
        edadf = st.session_state.df.copy()
        target = st.session_state.target

        # kdf = pd.read_csv("data.csv")
        # target = "Diagnosis"

        
        # Remove columns with all NaN values
        kdf = kdf.dropna(axis=1, how='all') 
        edadf = edadf.dropna(axis=1, how='all') 
        
        
        cat_cols = [col for col in kdf.columns if kdf[col].nunique() < 10]
        num_cols = [col for col in kdf.columns if col not in cat_cols]
        print("monuuu",target)
        print(kdf.columns)
        
        # separating target columns from categorical columns
        if target in cat_cols:
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
        edadf.drop(columns=newlist,axis=1,inplace=True)
        
        

        # label encoding the string categorical column 
        label_encoder = LabelEncoder()
        labelencodedcols =[]
        for column in kdf.columns:
            if kdf[column].dtype == 'object':  # Check if the column is of object type (string)
                labelencodedcols.append(column)
                kdf[column] = label_encoder.fit_transform(kdf[column])
                
        # # inversing label encoding for eda analysis
        # for column in labelencodedcols:
        #     edadf[column] = label_encoder.inverse_transform(kdf[column])
        st.session_state.edadf =edadf    
            





        if target in totalcols:
            st.text("Scroll to bottom to download the pre-processed dataset")
            # imputation of dataset
            with st.container(border=True):
                st.write(f"1st basic step involves understanding the features of data and to impute any NULL values. Here to impute any categorical column we have used *simple imputer with most frequent strategy* and impute numerical data we have used *simple imputer with median strategy*")
                st.write(np.round(kdf.describe(),1),height=400)
        
                imputeddf,oldreport,newreport = applyimputation(df=kdf)
                st.text(" ")
                col11,col12 = st.columns([1,1])
                with col11:
                    st.text("Before imputation percentage of null values")
                    st.dataframe(oldreport)
                with col12:    
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
                    
            st.subheader("3rd Step : ONE_HOT_ENCODING",anchor=False)       
            with st.container(border=True):
                st.write(f"Machine learning algorithms generally work better with numerical input rather than categorical data. One-hot encoding converts categorical variables into a format that can be used by these algorithms.By dropping the first category, we prevent multicollinearity in our model.Using dtype=np.int32 reduces the memory usage")
                onehotencodeddf = oneHotEncoding(df=nooutlierdf,numcols=num_cols,catcols=cat_cols)
                st.dataframe(onehotencodeddf.head())       
                    
                    
            # with st.container(border=True):
            #     st.subheader("Data Binning",anchor=False)
                
            #     st.write("*Data binning, also known as data discretization, is a technique used in data preprocessing that groups continuous values into discrete intervals or bins. This process helps in reducing the effects of minor observation errors and enhances the performance of models by simplifying the data.*")    
                
            #     st.write("__Here Age column has been binned into three bins Teenage , Adult and Senior__ ")
                
                # over or under sampling dataset
            st.subheader("4th Step : SMOTE",anchor=False)    
            with st.container(border=True):
                
                info = onehotencodeddf[target].value_counts()
                st.write("*SMOTE is an oversampling technique that creates synthetic samples of the minority class by interpolating between existing minority class instances.The goal of SMOTE is to balance the class distribution by generating synthetic examples rather than by duplicating existing ones, which helps prevent overfitting.*")   
                col1,col2 = st.columns([1,1])
                with col1:  
                    st.write("__Before oversampling__")
                    st.dataframe(info) 
                sampleddf = oversampling(df=onehotencodeddf,target=target)
                with col2:
                    st.write("__After Oversampling__")
                    newinfo = sampleddf[target].value_counts()
                    st.dataframe(newinfo) 
                
            # The encoded dataset is now scaled    
            st.subheader("5th Step : Min-Max Scaling",anchor=False)       
            
            with st.container(border=True):
                st.write("*Min-max scaling, also known as min-max normalization, transforms numerical features to a common scale, typically [0, 1]. It rescales features by shifting and scaling values to a specified range, where the minimum value of the feature becomes 0 and the maximum value becomes . Min-max scaling preserves the original distribution and relationships between data points.*") 
                st.text("After scaling the head of dataframe")
                report,scaleddf = getClassificationReport(df=sampleddf,target=target)    
                st.dataframe(scaleddf.head())    

                
                
                
            # Separate features and target variable
            X = scaleddf.drop(columns=[target])
            y = scaleddf[target]    
            
                # Sidebar for algorithm selection
            st.sidebar.header("Choose Classifier")
            classifier_name = st.sidebar.selectbox("Select Classifier", ("Logistic Regression", "Decision Tree", "Random Forest"))

            # Sidebar for classifier parameters
            def add_parameter_ui(clf_name):
                params = dict()
                if clf_name == "Logistic Regression":
                    params["C"] = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0)
                elif clf_name == "Decision Tree":
                    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20)
                    params["min_samples_split"] = st.sidebar.slider("Min Samples Split", 2, 20)
                elif clf_name == "Random Forest":
                    params["n_estimators"] = st.sidebar.slider("Number of Estimators", 10, 500,100)
                    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20)
                    params["min_samples_split"] = st.sidebar.slider("Min Samples Split", 2, 20)
                return params

            params = add_parameter_ui(classifier_name)
            
            X = scaleddf.drop(columns=[target])
            y = scaleddf[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # option to choose from the three classifier
            def get_classifier(clf_name, params):
                if clf_name == "Logistic Regression":
                    clf = LogisticRegression(C=params["C"])
                elif clf_name == "Decision Tree":
                    clf = DecisionTreeClassifier(max_depth=params["max_depth"], min_samples_split=params["min_samples_split"])
                elif clf_name == "Random Forest":
                    clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], min_samples_split=params["min_samples_split"])
                return clf

            clf = get_classifier(classifier_name, params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Display classification report
            st.text(" ")
            st.write("# Choose your classifier and set the parameters.")
            st.write(f"## Classification Report for {classifier_name}")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write(pd.DataFrame(report).transpose())

            st.text( "  ")
            if  not scaleddf.empty:
                filecontent = scaleddf.to_csv()
                st.download_button(label='Download the Pre-Processed Dataset',
                                data = filecontent,
                                type='primary',
                                file_name=f"preprocessed{target}.csv")
                
        else :
            st.warning("Enter correct Target column name (it is case sensitive)")        
        
    else:
        st.warning("PLEASE UPLOAD YOUR DATASET FIRST")    
        
        
        
if __name__ == "__main__":
    main()           
        
        
        