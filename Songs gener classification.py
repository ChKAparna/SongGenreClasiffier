#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score


# In[2]:


tracks = pd.read_csv(r"C:\Users\aparn\Downloads\song gener\fma-rock-vs-hiphop.csv")
music_features = pd.read_json(r"C:\Users\aparn\Downloads\song gener\echonest-metrics.json",precise_float=True)


# In[4]:


audio_data=music_features.merge(tracks[['genre_top', 'track_id']], on='track_id')


# In[5]:


audio_data.info()


# In[6]:


correlations = audio_data.corr()
correlations.style.background_gradient()


# In[7]:


features = audio_data.drop(['genre_top','track_id'],axis=1)
labels = audio_data['genre_top']


# In[9]:


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# In[10]:


pca = PCA()
pca.fit(scaled_features)
exp_variance = pca.explained_variance_ratio_
num_components = pca.n_components_


# In[11]:


fig, ax = plt.subplots()
ax.bar(range(num_components), exp_variance)
ax.set_xlabel('Principal Component #')


# In[12]:


cum_exp_variance = np.cumsum(exp_variance) #calculate the cumulative explained variance

fig, ax = plt.subplots()
ax.plot(cum_exp_variance)
ax.axhline(y=0.85, linestyle='--') #for considering features needed to explain 85% of the variance


# In[13]:


#finally perform PCA with the chosen number of components
pca = PCA(n_components=6, random_state=10)
pca.fit(scaled_features)
pca_projection = pca.transform(scaled_features)


# In[14]:


pca_projection.shape


# In[15]:


#split data into test and train data
X_train, X_test, y_train, y_test = train_test_split(pca_projection,labels,random_state=10)


# In[16]:


model_dt = DecisionTreeClassifier(random_state=10)
model_dt.fit(X_train,y_train)


# In[17]:


predictions_dt = model_dt.predict(X_test)
print("Decision Tree Classifier:", model_dt.score(X_test,y_test))


# In[18]:


#Train a Logistic Regression Model
model_lg = LogisticRegression(random_state=10)
model_lg.fit(X_train,y_train)


# In[19]:


predictions_lg = model_lg.predict(X_test)
print("Logistic Regression:", model_lg.score(X_test,y_test))


# In[20]:


report_dt = classification_report(y_test,predictions_dt)
report_lg = classification_report(y_test,predictions_lg)

print("Decision Tree Classifier: \n", report_dt)
print("Logistic Regression: \n", report_lg)


# In[23]:


#sns.countplot(audio_data['genre_top'], label = "Count") 


# In[24]:


# Subset only the hip-hop tracks, and then only the rock tracks
hip_hop = audio_data.loc[audio_data['genre_top'] == 'Hip-Hop']
rock = audio_data.loc[audio_data['genre_top'] == 'Rock']


# In[29]:


hip_hop.shape


# In[31]:


rock.shape


# In[32]:


rock = rock.sample(hip_hop.shape[0], random_state=10)


# In[33]:


#concatenate to create the balanced dataset
balanced_data = pd.concat([rock, hip_hop])
balanced_data.head()


# In[34]:


#features and labels for the new balanced data
features = balanced_data.drop(['genre_top', 'track_id'], axis=1) 
labels = balanced_data['genre_top']


# In[35]:


pca_projection = pca.fit_transform(scaler.fit_transform(features))
pca_projection.shape


# In[36]:


#split this balanced data into test and train
X_train, X_test, y_train, y_test = train_test_split(pca_projection, labels, random_state=10)


# In[37]:


#train decision tree on the balanced data
model_dt = DecisionTreeClassifier(random_state=10)
model_dt.fit(X_train,y_train)
predictions_dt = model_dt.predict(X_test)


# In[38]:


#train logistic regression on the balanced data
model_lr = LogisticRegression(random_state=10)
model_lr.fit(X_train,y_train)
predictions_lr = model_lr.predict(X_test)


# In[39]:


# compare the models
print("Decision Tree: \n", classification_report(y_test, predictions_dt))
print("Logistic Regression: \n", classification_report(y_test, predictions_lr))


# In[40]:


cv = KFold(n_splits=10, random_state=1, shuffle=True)


# In[41]:


scores = cross_val_score(model_dt, X_test,y_test, scoring='accuracy', cv=cv, n_jobs=-1)
print("Decision Tree Classifier Accuracy:", np.mean((scores)))


# In[42]:


scores = cross_val_score(model_lg, X_test,y_test, scoring='accuracy', cv=cv, n_jobs=-1)
print("Logistic Regression Accuracy:", np.mean((scores)))


# In[ ]:




