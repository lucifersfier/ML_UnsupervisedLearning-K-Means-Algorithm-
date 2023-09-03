#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Supervised Learning based on developed predictive model based on both input and output data Wheraeas
# Unsupervised Learning based on group and interpret data based only on input data

# Clustering is generally used for segmenting the customer

# Inertia tells us how far the points within a cluster are.
# The distance of a point from the centroid within the cluster intra cluster distance.
# 

# # Implementation of the K-means Algorithm 

# Problem Statement --> A marketing firm wants to launch the promotional campaign in different region of the country so in order to do that the firm needs to understand the diversity in population demography so that it can plan the promotional campaign accordingly

# In[8]:


data = pd.read_csv('Population_Data.csv')
data.head()


# In[9]:


data.info()


# Objective : - To segregate the regions into different groups

# In[13]:


numeric = ["Indians","Foreigners","Indian_Male","Indian_Female","Foreigners_Male","Foreigners_Female","Total Population"]

Now we are going to define a function for removing the commas from the numeric list elements 
# In[11]:


def cleaner(z):
    return z.replace(',','')


# In[14]:


for i in data[numeric]:
    data[i]=data[i].apply(cleaner)
data.head()


# In[16]:


data.info()


# Now because it is still showing non null object we are going to convert data type to numeric type explicitly

# In[19]:


data[numeric]=data[numeric].apply(pd.to_numeric)
data.info()


# In[22]:


#Verification of the integrity of data
data[["Indians",'Foreigners']].sum().sum() - data["Total Population"].sum()


# In[24]:


data[["Indian_Male","Indian_Female","Foreigners_Male","Foreigners_Female"]].sum().sum() - data["Total Population"].sum()


# we get the negative value which means total population > No. of males + No. of females 
# which means there are people who didnt identify thereselves as male or female

# In[27]:


MF_Sum = data['Indian_Male']+data["Indian_Female"]+data["Foreigners_Male"]+data["Foreigners_Female"]
data['other'] = data['Total Population'] - MF_Sum
data.head()


# In[30]:


data['Region'].nunique(),data['Office Location Id'].nunique()


# Region and Office Location Id are unique for every row and they donot helping in clustering so we will not consider then in clustering we will also drop the total Population from the data

# In[32]:


data1 = data.drop(columns=['Region','Office Location Id','Total Population'])
data1.head()


# In[33]:


from sklearn.preprocessing import Normalizer
norm = Normalizer()
columns = data1.columns
data1 = norm.fit_transform(data1)
data1 = pd.DataFrame(data1,columns=columns)
data1.head()


# In[35]:


from sklearn.cluster import KMeans
Kmeans = KMeans(n_clusters = 2)
Kmeans.fit(data1)
pred = Kmeans.predict(data1)


# In[36]:


pred,len(pred)


# In[37]:


Kmeans.inertia_


# Now to find out the right number of clusters --> by plotting a graph between the numbers of clusters and inertia
# Now we need to run the clustering algorithm several number of times and each time we eill be incrementally increasing the number of clusters and recording the corresponding inertia scores

# In[ ]:


SSE=[] #the empty list is to contain the inertia score whenever a clustering process is performed over the data 
for cluster in range(1,10):
    Kmeans=KMeans(n_jobs=-1, n_clusters=cluster)
    Kmeans.fit(data1)
    SSE.append(Kmeans.inertia_)


# In[ ]:


frame = pd.DataFrame({'Cluster':range(1,10),'SSE':SSE})


# In[ ]:


plt.figure(figsize = (12,6))
plt.plot((frame['Cluster'], frame['SSE'], marker='o'))
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')


# In[49]:


Kmeans=KMeans(n_clusters=3)
Kmeans.fit(data1)
pred = Kmeans.predict(data1)


# In[45]:


data1['cluster'] = pred


# In[46]:


def seg(str_x,str_y,clusters):
    x=[]
    y=[]
    for i in range(clusters):
        x.append(data1[str_x][data1['cluster']==i])
        y.append(data1[str_y][data1['cluster']==i])
        return x,y
def plot_clusters(str_x,str_y,clusters):
    plt.figure(figsize=(5,5),dpi=120)
    x,y=seg(str_x,str_y,clusters)
    
    for i in range(clusters):
        plt.scatter(x[i],y[i],label='cluster{}'.format(i))
        
    plt.xlabel(str_x)
    plt.ylabel(str_y)
    plt.title(str(str_x+"VS"+str_y))
    plt.legend()


# In[ ]:


plot_clusters('Indians','Foreigners',3)

