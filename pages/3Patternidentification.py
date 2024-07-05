import streamlit as st
import graphviz as gv
from io import StringIO
import streamlit as st
import plotly.express as px # type: ignore
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.optimize
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from p import applyimputation,testcoorelationship,getAllViolinPlots,remove_singlevariate_outliers,oneHotEncoding
from p import info,getallcatfig,describe_column,getallconfigs,gettopnfeatures,violin_plot,oversampling,getClassificationReport
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error




def load_data():
    return st.session_state.df


def main():
    
    # INITIAL CONFIGURATION
    kdf = load_data()
    tempdf = kdf
    st.write("## Dataset", tempdf.head())
    
    target = st.session_state.target
    if target == None:
        st.warning("Enter the Target Column")
    
    cat_cols = [col for col in tempdf.columns if tempdf[col].nunique() < 10]
    num_cols = [col for col in tempdf.columns if col not in cat_cols]
    cat_cols.remove(target)
    
    huicols = cat_cols + num_cols
    totalcols = tempdf.columns.tolist()
    noofcols = len(totalcols)
    
    # print(kdf.head())

    X = tempdf.drop(columns=[target])
    y = tempdf[target]

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.sidebar.title("Choose Algorithm",anchor=False)
    algorithm = st.sidebar.selectbox("Algorithm", ["K-Means", "DBSCAN"])
    
    if algorithm == "K-Means":
        st.sidebar.subheader("K-Means Parameters")
        col1,col2 = st.columns([1,1])
        x_feature = ""
        y_feature = ""
        
        with col1:
            x_feature = st.sidebar.selectbox("Select x variable",huicols)
                
        with col2:
            y_feature = st.sidebar.selectbox("Select y variable",huicols)
            
            
        n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
        
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X_scaled)
        tempdf['Cluster'] = kmeans.labels_
        
        st.write("## K-Means Clustering")
        # st.write()
        st.write("### Cluster Centers", kmeans.cluster_centers_)
        
        fig = px.scatter(tempdf, x=tempdf[x_feature], y=tempdf[y_feature], color='Cluster', color_continuous_scale='Viridis', title="Scatter Plot with Clusters")
        st.plotly_chart(fig)
        
        
    elif algorithm == "DBSCAN":
        st.sidebar.subheader("DBSCAN Parameters")
        col1,col2 = st.columns([1,1])
        x_feature = ""
        y_feature = ""
        
        with col1:
            x_feature = st.sidebar.selectbox("Select x variable",huicols)
                
        with col2:
            y_feature = st.sidebar.selectbox("Select y variable",huicols)
            
            
        eps = st.sidebar.slider("Epsilon", 0.1, 1.0, 0.5)
        min_samples = st.sidebar.slider("Min Samples", 1, 10, 5)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        tempdf['Cluster'] = dbscan.fit_predict(X_scaled)
        
        st.write("## DBSCAN Clustering")
        st.write(tempdf)
        
        fig = px.scatter(tempdf, x=tempdf[x_feature], y=tempdf[y_feature], color='Cluster', color_continuous_scale='Viridis', title="Scatter Plot with Clusters")
        st.plotly_chart(fig)
        
        

# Show the plot
       
        
        
if __name__ == "__main__":
    main()        