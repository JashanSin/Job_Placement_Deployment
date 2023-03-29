import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('Job_Placement_Data.csv')

df_clean = df.copy()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,8))   # This is a filter method useful to select features. With the help oh heatmap we will take only the subset of relevant feature means the feature that are relevent 
cor = df_clean.corr()      # we will select only those that has correlation more than 0.5 because closer value to 1 shows more positive coorelation
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

df_clean['hsc_subject'].unique()

ssc_board_type = pd.CategoricalDtype(categories = ['Others', 'Central'])
df_clean['ssc_board'] = df_clean['ssc_board'].astype(ssc_board_type)
gender_type = pd.CategoricalDtype(categories = ['M', 'F'])
df_clean['gender'] = df_clean['gender'].astype(gender_type)
hsc_board_type = pd.CategoricalDtype(categories = ['Others', 'Central'])
df_clean['hsc_board'] = df_clean['hsc_board'].astype(hsc_board_type)

undergrad_degree_type = pd.CategoricalDtype(categories = ['Sci&Tech', 'Comm&Mgmt', 'Others'])
df_clean['undergrad_degree'] = df_clean['undergrad_degree'].astype(undergrad_degree_type)
work_experience_type = pd.CategoricalDtype(categories = ['No', 'Yes'])
df_clean['work_experience'] = df_clean['work_experience'].astype(work_experience_type)
specialisation_type = pd.CategoricalDtype(categories = ['Mkt&HR', 'Mkt&Fin'])
df_clean['specialisation'] = df_clean['specialisation'].astype(specialisation_type)
status_type = pd.CategoricalDtype(categories = ['Placed', 'Not Placed'])
df_clean['status'] = df_clean['status'].astype(status_type)

# Replacing 'M' with'Male' and 'F' with 'Female'
df_clean['gender'].replace('M','Male' ,inplace = True)
df_clean['gender'].replace('F','Female' ,inplace = True)
df_clean = df_clean.drop(columns=['mba_percent','hsc_subject','hsc_board'])


from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
df_clean['gender'] = lab.fit_transform(df_clean['gender'])
df_clean['work_experience'] = lab.fit_transform(df_clean['work_experience'])
df_clean['specialisation'] = lab.fit_transform(df_clean['specialisation'])
#df_clean['status'] = lab.fit_transform(df_clean['status'])
df_clean['ssc_board'] = lab.fit_transform(df_clean['ssc_board'])
df_clean['undergrad_degree'] = lab.fit_transform(df_clean['undergrad_degree'])

import streamlit as st

import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.title("Job Placement Prediction")

st.sidebar.header('User Input Parameters')

def user_input_features():
  gender = st.sidebar.slider('gender', 0.0, 1.0, 0.6)
  ssc_percentage = st.sidebar.slider('ssc_percentage', 40.8,89.4, 67.3)
  ssc_board = st.sidebar.slider('ssc_board', 0.0, 1.0, 0.4)
  hsc_percentage = st.sidebar.slider('hsc_percentage', 37.0, 97.7, 66.3)
  degree_percentage = st.sidebar.slider('degree_percentage', 50.0,97.0, 66.3)
  undergrad_degree = st.sidebar.slider('undergrad_degree', 0.0,2.0 , 0.6)
  work_experience = st.sidebar.slider('work_experience', 0.0, 1.0, 0.3)
  emp_test_percentage = st.sidebar.slider('emp_test_percentage', 50.0, 98.0, 72.1)
  specialisation = st.sidebar.slider('specialisation', 0.0, 1.0, 0.4)

  user_input_data = {'gender': gender,
               'ssc_percentage': ssc_percentage,
               'ssc_board': ssc_board,
               'hsc_percentage': hsc_percentage,
               'degree_percentage': degree_percentage,
               'undergrad_degree': undergrad_degree,
               'work_experience': work_experience,
               'emp_test_percentage': emp_test_percentage,
               'specialisation': specialisation}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)


X = df_clean.loc[:, ['gender', 'ssc_percentage', 'ssc_board', 'hsc_percentage',
       'degree_percentage', 'undergrad_degree', 'work_experience',
       'emp_test_percentage', 'specialisation']]
y = df_clean['status']

# Encode the Target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Prediction
prediction = model.predict(df)

# Output
target_array = df_clean['status'].values
st.subheader('Prediction')
st.write(target_array[prediction])

prediction_probabilities = model.predict_proba(df)
#st.subheader('Class labels and their corresponding index number')
#st.write(target_array)

st.subheader('Prediction Probability')
st.write(prediction_probabilities)
