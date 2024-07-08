import streamlit as st
import graphviz as gv
from io import StringIO
import pandas as pd
import streamlit as st
import plotly.express as px # type: ignore
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from p import testcoorelationship,getAllViolinPlots
from p import getallcatfig,describe_column,getallconfigs



# generating coorelation matrix
def create_coorelation_matrix(df):
    corr_matrix = df.corr()

# Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))

    fig.update_layout(
        title='Correlation Heatmap',
        xaxis_title='Features',
        yaxis_title='Features'
    )

    return fig


# creating pie chart
def create_pie_charts(df, columns):
    figures = []
    
    for column in columns:
        if column in df.columns:
            fig = px.pie(df, names=column, title=f'Pie Chart for {column}')
            figures.append(fig)
        else:
            st.warning(f"Column '{column}' not found in DataFrame.")
    
    return figures

# CREATING HISTOGRAMS
def create_histograms(df, columns):
    figures = []
    
    for column in columns:
        if column in df.columns:
            fig = px.histogram(df, x=column, title=f'Histogram for {column}',
                               labels={column: column},
                               template='plotly_dark')
            fig.update_layout(
                title=dict(x=0.5),
                xaxis_title=column,
                yaxis_title='Count',
                bargap=0.1,         # Gap between bars of adjacent location coordinates
                bargroupgap=0  ,
            )
            # fig.show()
            figures.append(fig)
        else:
            st.warning(f"Column '{column}' not found in DataFrame.")
    
    return figures

# ANALYZING SKEWNESS OF DATA

def interpret_skewness(skewness):
    if skewness > 0.35:
        return "Strongly Right-skewed"
    elif skewness < -0.35:
        return "Strongly Left-skewed"

    else:
        return "Approximately Normally Distributed"
    
    # CREATING BOX PLOTS
def create_boxplots(df, columns):
    figures = []
    
    for column in columns:
        if column in df.columns:
            fig = px.box(df, y=column, title=f'Box Plot for {column}',
                         labels={column: column},
                         template='plotly_white')
            fig.update_layout(
                title=dict(x=0.5),
                yaxis_title=column,
            )
            figures.append(fig)
        else:
            st.warning(f"Column '{column}' not found in DataFrame.")
    
    return figures

# CREATING KDE PLOTS

def create_kde_plots(df, columns):
    figures = []
    
    for column in columns:
        if column in df.columns:
            fig, ax = plt.subplots()
            sns.kdeplot(data=df, x=column, ax=ax)
            ax.set_title(f'KDE Plot for {column}', color='white')  # Title color set to white
            ax.set_xlabel(column, color='white')  # X-axis label color set to white
            ax.set_ylabel('Density', color='white')  # Y-axis label color set to white
            ax.tick_params(axis='x', colors='white')  # X-axis tick color set to white
            ax.tick_params(axis='y', colors='white')  # Y-axis tick color set to white
            
            # Customize background color of plot area
            ax.set_facecolor('#2B2B2B')  # Dark gray background
            
            # Ensure spines (border lines) are not visible
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            
            figures.append(fig)
        else:
            st.warning(f"Column '{column}' not found in DataFrame.")
    
    return figures

# MAIN FUNCTION OF THIS PAGE

def main():
    
    st.title("Exploratory Data Analysis 📊🔍",anchor=False)
    
    if st.session_state.df is not None:
    
        # kdf = pd.read_csv("data.csv")
        # target = "Diagnosis"
        kdf = st.session_state.df
        edadf =st.session_state.edadf
        target = st.session_state.target
        

        # intial configuration
            
            
        cat_cols = [col for col in kdf.columns if kdf[col].nunique() < 10]
        num_cols = [col for col in kdf.columns if col not in cat_cols]
        
        if target in cat_cols:
            cat_cols.remove(target)
        
        totalcols = kdf.columns.tolist()
        totaledacols = edadf.columns.tolist()
        
        for col in totalcols:
            if col not in totaledacols:
                edadf[col] = kdf[col]
        st.dataframe(edadf.head())    
        
        
        noofcols = len(totalcols)

        # all plots 
        catboxfig = create_boxplots(kdf,cat_cols)        
        catunifig = create_pie_charts(edadf, cat_cols)
        numunifig = create_histograms(kdf,num_cols)
        numkdefig = create_kde_plots(kdf,num_cols)
        numboxfig = create_boxplots(kdf,num_cols)

        st.text("Co-relation Matrix of the dataset")
        fig = create_coorelation_matrix(kdf)
        st.plotly_chart(fig)

        st.subheader("Uni-Variate Analysis",anchor=False)
        with st.expander(" Click to show Univariate analysis of Categorical columns "):

            for i,fig in enumerate(catunifig) :
                if st.button(f"Show box plot of {cat_cols[i]} column",type='primary'):
                    st.plotly_chart(catboxfig[i])
                with st.container(border=True):
                    col1,col2 = st.columns([4,3])
                    
                    with col1:
                        st.plotly_chart(catunifig[i])
                    with col2:
                        st.write(f"Description of {cat_cols[i]}")
                        des = describe_column(df=kdf,column_name=cat_cols[i])
                        st.dataframe(des)         

    # UNIVARIATE ANALYSIS

        with st.expander(" Click to show Univariate Analysis of Numerical Columns "):

            for i,fig in enumerate(numunifig) :
                if st.button(f"Show KDE plot and Box plot for {num_cols[i]}",type='primary'):
                    col11,col22 = st.columns([1,1])
                    with col11:
                        st.plotly_chart(numkdefig[i])
                    with col22:
                        st.plotly_chart(numboxfig[i])    
                    
                with st.container(border=True):
                    skewness = 0
                    col1,col2 = st.columns([4,3])
                    
                    with col1:
                        st.plotly_chart(numunifig[i])
                    with col2:
                        st.write(f"Description of {num_cols[i]}")
                        des = describe_column(df=kdf,column_name=num_cols[i])
                        skewness = kdf[num_cols[i]].skew()
                        des['Skew'] = skewness
                        st.dataframe(des)   
                    s = interpret_skewness(skewness=skewness)
                    st.info(s)                          
                    
                    
                                #STARTING BI-VARIATE ANALYSIS WITH RESPECCT TO TARGET


        st.subheader("Bi-Variate Analysis",anchor=False)


        with st.expander(" Click to Bi-variate analysis of Categorical Columns "):
            catfigures = getallcatfig(kdf,cat_cols=cat_cols,target=target)

            for i,fig in enumerate(catfigures) :
                with st.container(border=True):
                    st.plotly_chart(catfigures[i])
        


        configs = getallconfigs(df=kdf,num_cols=num_cols,target=target)

        with st.expander(" Click to show Bi-variate analysis of Numerical Columns "):
            for i,fig in enumerate(configs) :
                with st.container(border=True):
                    col1,col2 = st.columns([4,3])

                    st.plotly_chart(fig)
            
                        
    # GETTING THE TOP N FEATURES USING VARIOUS METHODS
        with  st.container(border=True):
            st.subheader(f"Get most significant features from the dataframe with respect to label {target} using chi-squared, pearson-correlation and T-test  ",anchor=False)
            # topn = int(noofcols/3)
            st.write("") 
            # most = testcoorelationship(kdf,totalcols,target)
            # most.remove(target)
            # st.session_state.topnfeatures =  most
            most = st.session_state.topnfeatures.copy()
            st.write("Most Significant features are  :",most)
            # rdf = gettopnfeatures(n=topn,df= kdf,target=target) 
            # rdf.drop([0],inplace=True)
            # rdf = rdf.head(topn)
            # st.dataframe(data=rdf,height=500)
            # topncols = rdf.columns.tolist()
            st.text(' ')
            st.subheader(f"Visualizing Top feature Vs {target} in a Violoin Plot")
            allfigs = getAllViolinPlots(df=edadf,cols=most,target=target)
            for fig in allfigs:
                st.plotly_chart(fig)     
    else:
        st.warning("PLEASE UPLOAD YOUR DATASET FIRST")                                
            
        
if __name__ == "__main__":
    main()               