#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
from env import host, user, password
import os
from pydataset import data


# In[15]:


#Make a function named get_titanic_data that returns the titanic data from the codeup data science database as a pandas data frame. 
#Obtain your data from the Codeup Data Science Database.

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# In[16]:


def get_titanic_data():
    '''
    This function reads in the titanic data from the Codeup db
    and returns a pandas DataFrame with all columns.
    '''
    #create SQL query
    sql_query = 'SELECT * FROM passengers'
    
    #read in dataframe from Codeup db
    df = pd.read_sql(sql_query, get_connection('titanic_db'))
    
    return df



get_titanic_data().head()


# In[17]:


# Make a function named get_iris_data that returns the data from the iris_db on the codeup data science database as a pandas data frame. 
#The returned data frame should include the actual name of the species in addition to the species_ids. 
#Obtain your data from the Codeup Data Science Database.

def get_iris_data():
    '''
    This function reads in the iris data from the Codeup db
    and returns a pandas DataFrame.
    '''
    #create SQL query
    sql_query = 'SELECT * FROM measurements JOIN species USING (species_id)'
    
    #read in dataframe from Codeup db
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    
    return df



get_iris_data().head()


# In[18]:


# Once you've got your get_titanic_data and get_iris_data functions written, now it's time to add caching to them. 
#To do this, edit the beginning of the function to check for a local filename like titanic.csv or iris.csv. 
#If they exist, use the .csv file. If the file doesn't exist, then produce the SQL and pandas necessary to create a dataframe, 
#then write the dataframe to a .csv file with the appropriate name.

def cached_titanic_data(cached=False):
    '''
    This function reads in titanic data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in titanic df from
    a csv file, returns df.
    ''' 
    if cached == False or os.path.isfile('titanic_df.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = get_titanic_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('titanic_df.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('titanic_df.csv', index_col=0)
        
    return df


# In[19]:


titanic_df = cached_titanic_data(cached=False)
titanic_df.head()


# In[20]:


def new_iris_data():
    '''
    This function reads the iris data from the Codeup db into a df,
    writes it to a csv file, and returns the df.
    '''
    sql_query = """
                SELECT species_id,
                species_name,
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
                FROM measurements
                JOIN species
                USING(species_id)
                """
    df = pd.read_sql(sql_query, get_connection('iris_db'))
    df.to_csv('iris_df.csv')
    return df


# In[21]:


def cached_iris_data(cached=False):
    '''
    This function reads in iris data from Codeup database and writes data to
    a csv file if cached == False or if cached == True reads in iris df from
    a csv file, returns df.
    ''' 
    if cached == False or os.path.isfile('iris_df.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = get_iris_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('iris_df.csv')
    
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('iris_df.csv', index_col=0)
    
    return df


# In[22]:


iris_df = cached_iris_data(cached=False)
iris_df.head()


# In[ ]:





# In[ ]:




