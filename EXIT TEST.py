#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary liabraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Loading data
df=pd.read_csv('data_car.csv')


# In[3]:


#for showing first 5 columns of data under heading
df.head()


# In[4]:


df.shape


# In[5]:


#for summary of dataframes such as size and structure
df.info()


# In[6]:


#descriptive statistics of a dataframeobject
df.describe()


# In[7]:


#identifying null values
df.isna().sum()


# In[8]:


#to count the number of occurance in the Make ccolumn
df['Make'].value_counts()


# In[9]:


#number of unique values in msrp colum
df.MSRP.nunique()


# In[10]:


#to show the column labels
df.columns


# In[11]:


#columns with null values
no_columns =df[['Engine Fuel Type', 'Engine HP',
       'Engine Cylinders','Number of Doors','Market Category' ]]


# In[12]:


no_columns.isna().sum()


# In[13]:


#for summary of dataframes such as size and structure
df.info()


# In[14]:


#to find the date is skewed or not
freqgraph= df.select_dtypes(include ='float')
freqgraph.hist(figsize=(20,10))
plt.show()


# the data is skewed so we use median to fill the null values and for categorical column we use mode to fill null values

# In[15]:


#removing null values
df['Engine HP']=df['Engine HP'].fillna(df['Engine HP'].median())
df['Engine Cylinders']=df['Engine Cylinders'].fillna(df['Engine Cylinders'].median())
df['Number of Doors']=df['Number of Doors'].fillna(df['Number of Doors'].median())


# In[16]:


df.isna().sum()


# In[17]:


# Fill null values in the 'Engine Fuel Type' column with the most frequent fuel type
most_frequent_fuel_type = df['Engine Fuel Type'].mode()[0]
df['Engine Fuel Type'] = df['Engine Fuel Type'].fillna(most_frequent_fuel_type)


# In[18]:



# Fill null values in 'Market Category' column from 'Category' column
df['Market Category'].fillna(df['Market Category'].mode()[0], inplace=True)


# In[19]:


df.isna().sum()


# In[20]:


#creating a sctter plot to find models in different coloures
sns.FacetGrid(df, hue="Make", size=5)    .map(plt.scatter, "Popularity","MSRP")    .add_legend()


# In[21]:


#importing label encoder to convert categorical column to numerical column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 


# In[22]:


from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Define the categorical columns to be encoded
categorical_columns = ['Make', 'Model', 'Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Market Category', 'Vehicle Size', 'Vehicle Style']

# Iterate over the categorical columns and perform label encoding
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column].astype(str))

# Display the updated DataFrame
print(df)


# In[23]:


df.info()


# In[24]:


#split it into feature and target
y = df['MSRP']
X=df.drop(['MSRP'],axis=1)


# In[25]:


#split the data into testing and training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=42)


# # model training

# # 1.linear regression

# In[26]:


from sklearn import linear_model
lr=linear_model.LinearRegression()


# In[27]:


model=lr.fit(X_train,y_train)


# In[28]:


y_pred = model.predict(X_test)


# In[29]:


y_pred


# In[30]:


y_test


# In[31]:


from sklearn.metrics import r2_score
print('r2 score = ',r2_score(y_test,y_pred))


# In[32]:


from sklearn.metrics import mean_squared_error
print('MSE = ',mean_squared_error(y_test,y_pred))


# # LOGISTIC REGRESSION

# In[ ]:


#logistic regression modelfrom sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
model = clf.fit(X_train,y_train)


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
print('Accuracy = ',accuracy_score(y_test,y_pred))
print('Precision = ',precision_score(y_test,y_pred,average='macro'))
print('Recall = ',recall_score(y_test,y_pred,average='macro'))
print('f1 score = ',f1_score(y_test,y_pred,average='macro'))


# In[ ]:


y_pred =model.predict(X_test)


# In[ ]:


print (y_pred)


# In[ ]:


confusion_matrix(y_test,y_pred)


# # KNN

# In[ ]:


#importing knn algorithms
from sklearn.neighbors import KNeighborsClassifier
metric_k=[]
neighbors=np.arange(3,15)

for k in neighbors:
    classifier = KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=2)
    model = classifier.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    metric_k.append(acc)


# In[ ]:


metric_k


# In[ ]:


plt.plot(neighbors,metric_k,'o-')
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.grid()
plt.show()


# # svm

# In[ ]:


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Create an SVM classifier
clf = svm.SVC(kernel='linear')

# Train the SVM model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# # decision tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


dt_clf=DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train,y_train)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


y_pred_dt=dt_clf.predict(X_test)


# In[ ]:




