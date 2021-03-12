#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import random
import string
import re

# NLTK libraries
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

# Sci-kit libraries
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# # Data Loading

# In[2]:


data = pd.read_csv("./Dataset/training.1600000.processed.noemoticon.csv")
data = data.sample(n=20000) # using random sample of the actual data


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.columns = ['target','ids','Date','flag','user','text']


# In[6]:


data.info()


# In[7]:


# Dropping unnessary features

data.drop(['ids','Date','flag','user'],axis=1,inplace = True)


# In[8]:


data.target.value_counts()


# ## Data Cleaning

# In[9]:


punctuations = string.punctuation


# In[10]:


stop = stopwords.words('english')


# In[11]:


# appending punctuations in stopwords

punctuations = [char for char in punctuations]
for char in punctuations:
    stop.append(char)


# In[12]:


tokenizer = RegexpTokenizer(r'\w+') # only aplhabets 
ps = PorterStemmer()


# In[13]:


def cleanWords(text):
    
    # lower the text message
    text = text.lower()
    
    # remove links
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',text)
    
     # remove usernames
    text = re.sub('@[^\s]+','',text) 
    
     # remove additional whitespaces
    text = re.sub('[\s]+', ' ', text)
    
    # Regex tokenizer
    text = tokenizer.tokenize(text)
    
    # Stopwords removal and Stemming using porter stemmer
    meaningful = [ps.stem(word) for word in text if not word in stop]
        

    return ' '.join(meaningful)


# In[14]:


key = data['text'].keys()


# In[15]:


# Cleaning all texts in dataFrame

for i in key:
    data['text'][i] = cleanWords(data['text'][i])
    


# In[16]:


data.head(7)


# ##  Data Splitting

# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train , X_test , Y_train, Y_test = train_test_split(data, data['target'],test_size=0.2,random_state=0)


# In[19]:


X_train.shape , X_test.shape


# In[20]:


X_train.head(5), Y_train.head(4)


# ## Creating vocab and data formatting

# In[22]:


from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer


# In[23]:


tdidf = TfidfVectorizer(analyzer='word', max_features=2000, max_df = 0.8, ngram_range=(1,1))
X_train_vectorized = tdidf.fit_transform(X_train.text)
X_test_vectorized = tdidf.transform(X_test.text)


# In[24]:


X_train_vectorized.shape, X_test_vectorized.shape


# # Model Selection

# ### <font color= 'red'>Logistic Regression </font>

# In[25]:


logreg = LogisticRegression(C = 2.1, solver='liblinear', multi_class='auto')
logreg.fit(X_train_vectorized, Y_train)
Y_pred_lr = logreg.predict(X_test_vectorized)

cf_lr = classification_report(Y_pred_lr,Y_test)
score_lr = accuracy_score(Y_pred_lr,Y_test)

print(cf_lr)
print("Accuracy " ,score_lr)


# ### <font color= 'red'>SVC </font>

# In[26]:


svc = SVC()
svc.fit(X_train_vectorized, Y_train)
Y_pred_svc = svc.predict(X_test_vectorized)

cf_svc = classification_report(Y_pred_svc,Y_test)
score_svc = accuracy_score(Y_pred_svc,Y_test)
print(cf_svc)
print("Accuracy " , score_svc)


# ### <font color= 'red'>Random Forest Classifier </font>

# In[27]:


rf = RandomForestClassifier()
rf.fit(X_train_vectorized, Y_train)
Y_pred_rf = rf.predict(X_test_vectorized)

cf_rf = classification_report(Y_pred_rf,Y_test)
score_rf = accuracy_score(Y_pred_rf,Y_test)

print(cf_rf)
print("Accuracy " ,score_rf)


# ### <font color= 'red'>Decision Tree Classifier </font>

# In[28]:


dt = DecisionTreeClassifier()
dt.fit(X_train_vectorized, Y_train)
Y_pred_dt = dt.predict(X_test_vectorized)

cf_dt = classification_report(Y_pred_dt,Y_test)
score_dt = accuracy_score(Y_pred_dt,Y_test)
print(cf_dt)
print("Accuracy " ,score_dt)


# In[ ]:





# In[ ]:




