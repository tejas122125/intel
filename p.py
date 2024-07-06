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
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder





catimputer = SimpleImputer(strategy='most_frequent')
numimputer = SimpleImputer(strategy='median')

ohe = OneHotEncoder(drop='first',sparse_output=False,dtype=np.int32)

def scatter_plot(df,column1,column2="Diagnosis"):
    fig = px.scatter(
        df,
        x=column1,
        y=column2,
        color=column2,
        title=f'Scatter Plot of {column1} vs {column2}',
        labels={column1: column1,column2: column2},
        color_discrete_map={0: 'green', 1: 'red'},
        template='plotly_dark'
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8), selector=dict(mode='markers'))
    # fig.show()
    return fig
# violin plot

def violin_plot(df,column1,column2):
    fig = px.violin(
        df,
        x=column2,
        y=column1,
        color=column2,
        title=f'Violin Plot of {column1} by {column2}',
        labels={column1: column1, column2: column2},
        color_discrete_map={0: 'green', 1: 'red'},
        template='plotly_dark'
    )
    # fig.show()
    return fig
def getAllViolinPlots(df,cols,target):
    allfigs = []
    for col in cols:
        allfigs.append(violin_plot(df=df,column1=col,column2=target))
    return allfigs    


# Histogram
def histogram(df,column1):
    fig = px.histogram(df, x=column1, title=f'Histogram of {column1}')
    # fig.
    return fig

# Pie Chart
def pie_chart(df,column1):
    fig = px.pie(df, names='Category', title='Pie Chart of Categories')
    return fig

# Streamlit app layout
# st.title("Plotly Visualizations in Streamlit")

# # Scatter plot
# st.header("Scatter Plot")
# scatter_fig = scatter_plot(df)
# st.plotly_chart(scatter_fig)

# # Histogram
# st.header("Histogram")
# histogram_fig = histogram(df)
# st.plotly_chart(histogram_fig)

# # Pie chart
# st.header("Pie Chart")
# pie_fig = pie_chart(df)
# st.plotly_chart(pie_fig)
# various tesst 
# to test two continous value datas
def pearson_coorelation(df,x,y):
    correlation, p_value = pearsonr(df[x], df[y])
    return p_value


# to test categorical data
def chi_square_test(df,x,y):
    # Create a contingency table
    contingency_table = pd.crosstab(df[x], df[y])
    # print(contingency_table)
    chi2, p, dof, ex = chi2_contingency(contingency_table)
    return  p
#  to test a continous data with categorical data
def t_test(df,x,y):
    diagnosis_0 = df[df[y] == 0][x]
    diagnosis_1 = df[df[y] == 1][x]
    t_stat, p_value = ttest_ind(diagnosis_0, diagnosis_1)
    return p_value

def info(df):
    return df.info()


def oneHotEncoding(df,numcols,catcols):
    # Identify string columns in which are not required
    string_columns = df.select_dtypes(include=['object']).columns.tolist()
    newlist = []
    for col in  string_columns:
        if col in numcols:
            newlist.append(col)
            
    newdf = df
    
    allcols = newdf.columns.to_list()        
    # print(feature_names)
    # allcol = allcol + feature_names
    prevdf =  newdf.drop(catcols,axis=1)
    #  removing unrequired string columns from dataset
    prevdf = prevdf.drop(newlist,axis =1)
    
    newdf =ohe.fit_transform(newdf[catcols])
    feature_names = ohe.get_feature_names_out(catcols)
    newdf1 = pd.DataFrame(newdf,columns=feature_names)
    
    finaldf = pd.concat([newdf1,prevdf],axis = 1)
    return finaldf
    
    #FUNCTION RETURNING TOP N  FEATURES WITH RESPECT TO TARGET COLUMN  USING VARIOUS TEST SCORES 
def testcoorelationship(df,cols ,target):
    mostsignificant = []
    totalcols = df.columns.tolist()
    for x in cols:
        if x in totalcols:
            x_unique = df[x].nunique()
            y_unique = df[target].nunique()
            p_value = 0

            if x_unique > 5 & y_unique >5:
                # print("using pearson coorelation")
                p_value = pearson_coorelation(df,x,target)
            elif x_unique < 5 & y_unique < 5:
                # print("using chisquared test")
                p_value = chi_square_test(df,x,target)
            elif x_unique >5 & y_unique < 5:
                # print("using t test")
                p_value = t_test(df,x,target)
                
            else:
                print("using coorelation")
                p_value = pearson_coorelation(df,x,target)    

            if p_value < 0.06:
                # print("There is a significant relationship between age and diagnosis (p < 0.05).")
                # testresult = f"There is a significant relationship between {x} and {y} ({p_value} < 0.05)."
                mostsignificant.append(x)
    return mostsignificant    
    
    
    
# kdf = pd.read_csv('kidney.csv')
# kdf.drop(columns = ['PatientID', 'DoctorInCharge'], inplace = True)
# cat_cols = [col for col in kdf.columns if kdf[col].nunique() < 6]
# num_cols = [col for col in kdf.columns if col not in cat_cols]

# cat_cols.remove("Diagnosis")


# xlabel_ca =  {
#     "Gender": "Gender(0: Male, 1: Female)",
#     "Ethnicity": "Ethnicity(0: Caucasian, 1: African American, 2: Asian, 3: Other)",
#     "EducationLevel": "EducationLevel(0: None, 1: High School, 2: Bachelor's, 3: Higher)",
#     "Smoking": "Smoking(0: No, 1: Yes)",
#     "FamilyHistoryKidneyDisease": "FamilyHistoryKidneyDisease(0: No, 1: Yes)",
#     "FamilyHistoryHypertension": "FamilyHistoryHypertension(0: No, 1: Yes)",
#     "FamilyHistoryDiabetes": "FamilyHistoryDiabetes(0: No, 1: Yes)",
#     "PreviousAcuteKidneyInjury": "PreviousAcuteKidneyInjury(0: No, 1: Yes)",
#     "UrinaryTractInfections": "UrinaryTractInfections(0: No, 1: Yes)",
#     "ACEInhibitors": "ACEInhibitors(0: No, 1: Yes)",
#     "Diuretics": "Diuretics(0: No, 1: Yes)",
#     "Statins": "Statins(0: No, 1: Yes)",
#     "AntidiabeticMedications": "AntidiabeticMedications(0: No, 1: Yes)",
#     "Edema": "Edema(0: No, 1: Yes)",
#     "HeavyMetalsExposure": "HeavyMetalsExposure(0: No, 1: Yes)",
#     "SocioeconomicStatus": "SocioeconomicStatus(0: Low, 1: Middle, 2: High)",
#     "OccupationalExposureChemicals": "OccupationalExposureChemicals(0: No, 1: Yes)",
#     "WaterQuality": "WaterQuality(0: Good, 1: Poor)"
# }
# PARSING THE COLUMNS INFORMATION INTO A DICTIONARY
# def parse_column_descriptions(input_string):
#     # Split the input string by periods to get individual column descriptions
#     single_line_string = ' '.join(line.strip() for line in input_string.splitlines())
#     descriptions = single_line_string.split('. ')
#     # print(len(descriptions))
#     # Initialize an empty dictionary to store the results
#     result = {}
    
#     # Iterate over the descriptions
#     for description in descriptions:
#         # Split each description by the first occurrence of ': ' to separate the column name from its description
#         parts = description.split(': ', 1)
#         if len(parts) == 2:  # Ensure there are exactly two parts
#             column_name = parts[0].strip()
#             column_description = parts[1].strip()
#             # Add the column name and description to the dictionary
#             result[column_name] = column_description
    
#     return result




paperbgcolor  = '#372694'
plotbgcolor = "#372694"

def plot_percentage_stacked_bar_plotly(df, feature, label):
    
    # Calculate counts
    counts = df.groupby([feature, label]).size().unstack(fill_value=0)

    # Normalize counts to percentages
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    # print(percentages)
    # Create traces for each diagnosis label
    traces = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for diagnosis3,color in zip(percentages.columns,colors):
        traces.append(go.Bar(
            x=percentages.index,
            y=percentages[diagnosis3],
            name=f'{label} = {diagnosis3}',
            marker_color=color,
            opacity=0.8,
            text=[f'{p:.1f}%' for p in percentages[diagnosis3]],  # Add percentages as text
            textposition='inside',
            textfont_color='white', 
        ))
    # Create layout
    layout = go.Layout(
        barmode='stack',
        title=f'<b>Percentage Stacked Bar Chart for {feature}</b>',
        xaxis=dict(title=f'<b>{feature}</b>',linewidth=2,color = 'white',gridcolor= '#F8F2B7'),
        yaxis=dict(title=f'<b>Percentage</b>',linewidth=2,color = 'white',gridcolor = '#F8F2B7'),
        legend=dict(title=f'<b>{label}</b>'),   
        paper_bgcolor=paperbgcolor , 
        plot_bgcolor=plotbgcolor,
          # Set figure width
        height=425 ,
        
        font={'color':'white'} 
    )
    # Create figure and plot
    fig = go.Figure(data=traces, layout=layout)
    return fig
    
    # fig.show()

# Plot percentage stacked bar charts for each categorical feature
def getallcatfig(df,cat_cols,target):
    allcategoriesfigures  =[] 
    for feature in cat_cols:
        fig =  plot_percentage_stacked_bar_plotly(df, feature, target)
        allcategoriesfigures.append(fig)
    return allcategoriesfigures   
   
   
def describe_column(df, column_name):
    
    """
    Returns descriptive statistics for a specific column in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to describe.

    Returns:
    pandas.Series: A series containing the descriptive statistics of the column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")
    nullcount = df[column_name].isnull().sum()
    df_x = np.round(df[column_name].describe(),1)
    df_x['nullcount']  = nullcount
    return df_x


# Define custom colors
# DRAWING HISTOGRAM OF ALL COLUMNS VS TARGET COLUMN
def continuousdata(df,column_name,target):
  color_discrete_map = {0: '#267E0A', 1: '#A31010'}
  
  colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
  # Create the Plotly histogram with KDE
  fig = px.histogram( 
      df,
      x=column_name,
      color=target,
      nbins=10,
      opacity=0.85,
    #   color_discrete_map=color_discrete_map,
      title=f'{column_name} Distribution by {target}',
  )

  # Update layout for better appearance
  fig.update_layout(
      xaxis_title=column_name,
      xaxis = dict(linewidth = 2,gridcolor='#F8F2B7'),
      yaxis = dict(linewidth = 2,gridcolor = '#F8F2B7'),
      yaxis_title='Count',
      legend_title=target,
          paper_bgcolor=paperbgcolor , 
          plot_bgcolor=plotbgcolor,
            # Set figure width
          height=425 ,
              bargap=0.1,         # Gap between bars of adjacent location coordinates
      bargroupgap=0  ,
          font={'color':'white'} 
  )
  return fig

# RETURNING THE PLOTTED FIGURES
def getallconfigs(df,num_cols,target):
    numfigs = []
    for col in num_cols:
        numfigs.append(continuousdata(df,col,target))
    return numfigs    

  
#   GETTING THE TOP N FEATURES FOR THE DATASET WITH RESPECT TO TARGET COLUMN
def gettopnfeatures(n,df,target):
#  obtaining top 20  related features
    corr_matrix =df.corr()
    # corr_matrix.style.background_gradient(cmap='coolwarm')
    diag_corr = []

    for i in range(len(corr_matrix.index)):
        diag_corr.append([corr_matrix.index[i], corr_matrix[target][i]])

    diag_corr_df = pd.DataFrame(diag_corr, columns=["Features", f"CorrWith{target}"])
    diag_corr_df[f"CorrWith{target}Absolute"] = diag_corr_df[f"CorrWith{target}"].abs()
    sorted_diag_corr_df = diag_corr_df.sort_values(f"CorrWith{target}Absolute", ascending=False).reset_index(drop=True)
    sorted_diag_corr_df[sorted_diag_corr_df["Features"]!=target].head(n)
    return sorted_diag_corr_df

# def getPca3dfig(df):
#     # for fun 
#     columns = df.columns.to_list()
#     columns.remove('Diagnosis')
#     x= df[columns]
#     y=df['Diagnosis']
#     x=StandardScaler().fit_transform(x)
    
    
#     pca = PCA(n_components=3)
#     principal_components = pca.fit_transform(x)
    
#     pca_df = pd.DataFrame(data=principal_components,columns=['PC1','PC2','PC3'])
#     pca_df['Diagnosis'] = y
#     fig = px.scatter_3d(
#     pca_df,
#     x='PC1',
#     y='PC2',
#     z='PC3',
#     color=pca_df['Diagnosis'].astype(str),
#     title='3D PCA of Multidimensional Data',
#     labels={'PC1': 'PC1', 'PC2': 'PC1', 'PC3': 'PC3'},
#     color_discrete_map={'0': 'green', '1': 'red'}
# )
#     return fig


# FUNCTION TO REMOVE SINGLE VARIATE OUTLIER USING IQR METHODS
def remove_singlevariate_outliers(df, columns, lower_percentile=0.01, upper_percentile=0.75):

    for column in columns:

        # Calculate the lower and upper percentile values
        lower_bound = df[column].quantile(lower_percentile)
        upper_bound = df[column].quantile(upper_percentile)
        iqr = lower_bound - upper_bound
        upper_limit = upper_bound + 1.5 * iqr
        lower_limit = lower_bound - 1.5 * iqr        
        new_df_cap = df.copy()

        new_df_cap[column] = np.where(
            new_df_cap[column] > upper_limit,
            upper_limit,
            np.where(
                new_df_cap[column] < lower_limit,
                lower_limit,
                new_df_cap[column]
            )
        )
        
    return new_df_cap

# FUNCTION TO REMOVE MULTIVARIATE OUTLIER USING MAHALANOBIS DISTANCE METHOD (NOT USED)
# mahalanobis outlier removal
# Compute Mahalanobis distance
def multivariate_outlier_removal(orignal = None,x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = scipy.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    orignal["mahala"]  = mahal.diagonal()
    outlier = orignal['mahala'].quantile(0.99)
    new_df = orignal[orignal['mahala']<= outlier]
    new_df = new_df.drop('mahala',axis=1)
    return new_df
    
    
   # REMOVING ALL KIND OF OUTLIER USING ABOVE TWO METHODS 
def remove_all_outlier(df,singlevariate = ['Age','BMI'],col_for_multivariate_outlier = ['MedicationAdherence','HealthLiteracy','AlcoholConsumption','PhysicalActivity'] ):
  
    fildf = remove_singlevariate_outliers(df,singlevariate)
    fildf.shape
    df_x = fildf[col_for_multivariate_outlier]#.head(500)
    new_df = multivariate_outlier_removal(orignal=fildf,x=df_x, data=df[col_for_multivariate_outlier])
    new_df.shape
    return new_df
                   
# FUNCTION FOR DATABINING (NOT USED)

# def databining(col,cut_points,labels = None):
#     minval = col.min()
#     maxval = col.max()
#     break_points =[ minval]+cut_points+[maxval]
#     break_points.sort()
#     if not labels :
#         labels = range(len(cut_points)+1)
#     colBin = pd.cut(col,bins = break_points,labels=labels,include_lowest=True)
#     return colBin    


# # bindf = new_df.copy()
# # cut_points = [18,40]
# # labels = ['Teenage','Adult','Senior']
# # bindf['age_category'] = databining(col=bindf['Age'],cut_points=cut_points,labels=labels)
# # bindf1 = bindf.drop('Age',axis=1)



# Apply SMOTE to the feature and target datasets
def oversampling(df,target):
    X = df.drop(target, axis=1)
    y = df[target]
# Initialize SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    # Combine the resampled features and target into a DataFrame
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target] = y_resampled
    print("Original DataFrame:")
    print(df.shape)
    print("\nResampled DataFrame:")
    print(df_resampled.shape)
    return df_resampled



# Here's a function that takes a DataFrame and returns a scaled DataFrame using Min-Max scaling:

def minmaxscaling(df,target):
    
    scaler = MinMaxScaler()
    
    x = df.drop(target, axis=1)
    xcols = x.columns
    y = df[target]
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        
    x_scaled = scaler.fit_transform(x)
    x_scaleddf = pd.DataFrame(x_scaled,columns=xcols)
    x_scaleddf[target] = y
    finaldf = x_scaleddf
    return finaldf

    
def getClassificationReport(df,target):    
    
    # df_resamp = oversampling(df,target)
    # print(df_resamp[target].value_counts())  
    df_scaled = minmaxscaling(df,target=target) 
    x = df.drop(target, axis=1)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
       
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print the classification report
    report = classification_report(y_test, y_pred)
    # print(type(report))
    
    return report,df_scaled


# FUNCTION FOR IMPUTING NUMERICAL COLUMNS
def imputingcat_num(df,zero,cat,num):
    allcol = df.columns.to_list()
    transformer = ColumnTransformer(transformers=[
    ('tnf1',catimputer,cat),
    ('tnf2',numimputer,num),
],remainder='passthrough')
    df_final = pd.DataFrame(transformer.fit_transform(df),columns=allcol)
    print(df_final.shape)
    return df_final
    

# FUNCTION FOR IMPUTING CATEGORICAL COLUMNS
def categorize_columns_by_null(df):
    cat_cols = [col for col in df.columns if df[col].nunique() < 10]
    num_cols = [col for col in df.columns if col not in cat_cols]
    
    total_rows = len(df)
    null_percentages = df.isnull().sum() / total_rows * 100
    nonzerocat = []
    nonzeronum = []
    zero= []
    

    for col in df.columns:
        null_percentage = null_percentages[col]
        if null_percentage == 0:
            zero.append(col)
        else:
            if col in cat_cols :
                nonzerocat.append(col)
            else:
                nonzeronum.append(col)    
    
    return zero, nonzerocat,nonzeronum


def applyimputation(df):
    zero,nonzerocat,nonzeronum = categorize_columns_by_null(df=df)
    print(nonzeronum)
    imputeddf = imputingcat_num(df=df,zero=zero,cat=nonzerocat,num=nonzeronum)
    newreport = pd.DataFrame(imputeddf.isnull().mean()*100)
    oldreport = pd.DataFrame(df.isnull().mean()*100)
    
    return imputeddf,oldreport,newreport
    
    
