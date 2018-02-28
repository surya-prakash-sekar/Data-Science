
# coding: utf-8

# In[2]:

from __future__ import division
from numpy import *
import pandas as pd
import numpy as np
import random
import math as m
import collections
import sys 
sys.setrecursionlimit(2000)
euc_label={}
elk_label={}
cos_label={}
eq1_label={}
eq2_label={}
city_label={}

def euclidean_distance(df,cen,rows,k,cols):
    x1_all=np.zeros(shape=(rows,k))
    x2_all=np.zeros(shape=(k,cols))
    new_cent=np.zeros(shape=(k,cols))
    cnts=np.zeros(shape=(k,1))
    
    for i in range(rows):
        for j in range(k):
            x1_all[i][j]=np.absolute(np.sqrt(np.sum((df[i, :] - cen[j, :]) ** 2)))

    #print 'x1_all :' ,'\n'             
    #print(x1_all) 

    for i in range(rows): 
        euc_label[i]=list(x1_all.argsort()[i][:1])
    
    #print("euc label") 
    #print(euc_label)
    
    for i in range(k):
        for x in range(rows):
            if [i] == euc_label[x]:
                cnts[i]=cnts[i]+1
                
    #print 'counts'
    #print cnts
  
    for z in range(k):    
        for i in range(rows):
            if [z] == euc_label[i]:
                x2_all[z]=x2_all[z]+df[i]

    #print'x2_all','\n'
    #print x2_all
    
    for i in range(k):
        new_cent[i]=np.absolute(x2_all[i]/cnts[i])
     
    for i in range(k):
        for j in range(cols):
            new_cent[i][j]=round(new_cent[i][j],1)
            
    return new_cent

def cityblock_distance(df,cen,rows,k,cols):
    x1_all=np.zeros(shape=(rows,k))
    x2_all=np.zeros(shape=(k,cols))
    new_cent=np.zeros(shape=(k,cols))
    cnts=np.zeros(shape=(k,1))
    
    for i in range(rows):
        for j in range(k):
            x1_all[i][j]=np.absolute(np.sum((df[i, :] - cen[j, :])))

    #print 'x1_all :' ,'\n'             
    #print(x1_all) 

    for i in range(rows): 
        city_label[i]=list(x1_all.argsort()[i][:1])
    
    #print("euc label") 
    #print(euc_label)
    
    for i in range(k):
        for x in range(rows):
            if [i] == city_label[x]:
                cnts[i]=cnts[i]+1
                
    #print 'counts'
    #print cnts
  
    for z in range(k):    
        for i in range(rows):
            if [z] == city_label[i]:
                x2_all[z]=x2_all[z]+df[i]

    #print'x2_all','\n'
    #print x2_all
    
    for i in range(k):
        new_cent[i]=np.absolute(x2_all[i]/cnts[i])
     
    for i in range(k):
        for j in range(cols):
            new_cent[i][j]=round(new_cent[i][j],1)
            
    return new_cent


def elkans(df,cen,rows,k,cols,lb_all,ub_all,elk_label):
    
    old_cent=np.zeros(shape=(k,cols))
    test=np.zeros(shape=(1,cols))
    old_cent=cen
    
    print 'elkans was called with centroid' '\n'
    print old_cent 
    print '--------------------------------' '\n'
    
    d_c=np.zeros(shape=(k,k))
    s=np.zeros(shape=(k,1))   
    shift=np.zeros(shape=(k,1))

    new_cent=np.zeros(shape=(k,cols))
    x2_all=np.zeros(shape=(k,cols))
    cnts=np.zeros(shape=(k,1))     
        
    # Finding d(c,c')
            
    for i in range(k):
        for j in range(k):
            if i!=j:
                d_c[i][j]=np.absolute(np.sqrt(np.sum((cen[i,:] - cen[j,:]) ** 2))) 
    #print d_c 
        
    # Finding s(j) - half the distance between a Centroid and its closest centroid
        
    for i in range(k): 
        te=d_c.argsort()[i][:2]
        s[i]=d_c[i][te[1]]/2
        #print te
    #print 's'
    #print s
    #print '--------------------------------' '\n'
        
    # Algorithm Implementation
        
    for i in range(rows):  

        if ub_all[i] <= s[list(elk_label[i]).pop()]:
            continue
            
        r=1; # r is set to true       
            
        for j in range(k): #max(lb_all[i][j],s[j])
            z=max(lb_all[i][j],np.sqrt(np.sum((cen[list(elk_label[i]).pop(),:] - cen[j,:]) ** 2))/2)
                
            if j==list(elk_label[i]).pop(): 
                continue 
                    
            if ub_all[i]<=z:
                continue
                    
            if r:
                ub_all[i]=np.absolute(np.sqrt(np.sum((df[i,:]-cen[list(elk_label[i]).pop(),:])** 2))) #abs(df[i][j]-cen[elk_label[i]])
                r=0; # r is set to false
                    
                if ub_all[i]<=z:
                    continue
                        
            lb_all[i][j]=np.absolute(np.sqrt(np.sum((df[i,:]-cen[j,:])** 2))) #abs(df[i][j]-cen[j])
                    
            if lb_all[i][j]<ub_all[i]:
                elk_label[i]=[j]
    
    # Calculating the new centroids
    
    for i in range(k):
        for x in range(rows):
            if [i] == elk_label[x]:
                cnts[i]=cnts[i]+1
                
    print 'counts'
    print cnts
    print '--------------------------------' '\n'
  
    for z_1 in range(k):    
        for i in range(rows):
            if z_1 == list(elk_label[i]).pop():
                x2_all[z_1]=x2_all[z_1]+df[i]
        
    for i in range(k):
        new_cent[i]=x2_all[i,:]/cnts[i]
     
    
    # Handling the cases where centroids turn up to 0
    
    where_are_NaNs = isnan(new_cent)
    new_cent[where_are_NaNs] = 0  
         
    for i in range(k):
        for j in range(cols):
            new_cent[i][j]=round(new_cent[i][j],1) 
  
    for o in range(k):
        if np.array_equal(new_cent[o],test[0]):
            m = max(cnts)
            m1=[i for i, j in enumerate(cnts) if j == m]        
            new_cent[o]=old_cent[o]  # alternative approach we can also assign somewhere close to centroid having most elements new_cent[m1]+.1 
            #new_cent[o]=new_cent[m1]+0.1
    #Calculating the centroid shift
        
    for i in range(k):
        shift[i]=np.absolute(np.sqrt(np.sum((new_cent[i] - cen[i]) ** 2)))
        
        #Updating the upper and lower bounds
    for i in range(rows):
        ub_all[i]=ub_all[i]+shift[list(elk_label[i]).pop()]#np.absolute(np.sum(ub_all[i],shift[elk_label[i]]))
        for j in range(k):
            lb_all[i][j]=max(0,lb_all[i][j]-shift[j])       
             
    print 'new cent'
    print new_cent   
    print '--------------------------------' '\n'
    
    return new_cent

def cosine_distance(df,cen,rows,k,cols):
    x1_all=np.zeros(shape=(rows,k))
    x2_all=np.zeros(shape=(k,cols))
    new_cent=np.zeros(shape=(k,cols))
    cnts=np.zeros(shape=(k,1))
    
    for i in range(rows):
        for j in range(k):
            x1_all[i][j]=(1 - (np.dot(df[i, :], cen[j, :]) / (np.linalg.norm(df[i, :]) * np.linalg.norm(cen[j, :]))))

    for i in range(rows): 
        cos_label[i]=list(x1_all.argsort()[i][:1])
    
    for i in range(k):
        for x in range(rows):
            if [i] == cos_label[x]:
                cnts[i]=cnts[i]+1
  
    for z in range(k):    
        for i in range(rows):
            if [z] == cos_label[i]:
                x2_all[z]=x2_all[z]+df[i]
    
    for i in range(k):
        new_cent[i]=np.absolute(x2_all[i]/cnts[i])
     
    for i in range(k):
        for j in range(cols):
            new_cent[i][j]=round(new_cent[i][j],1)
            
    return new_cent
   
def new_distance1(df,cen,rows,k,cols):
                
    x2_all=np.zeros(shape=(k,cols))
    new_cent=np.zeros(shape=(k,cols))
    cnts=np.zeros(shape=(k,1))
    x3=[]
    x3_all=np.zeros(shape=(rows,k))
  
    for i in df:
        for j in cen:
            term = 0.0
            termx = 0.0
            termy = 0.0
            for p, l in zip(i, j):
                if (p > l):
                    difference = p - l
                    termx += difference
                elif (p < l):
                    difference = l - p
                    termy += difference
            termsqr = np.square(termx) + np.square(termy)
            term = np.sqrt(termsqr)
            x3.append(term)
            
    #print x3
    #print '\n'
    #print len(x3)
            
    x3= np.reshape(x3,(rows,k)) 
    
    for i in range(rows):
        for j in range(k):            
            x3_all[i][j]=x3[i][j]

    for i in range(rows): 
        eq1_label[i]=list(x3_all.argsort()[i][:1])
    
    for i in range(k):
        for x in range(rows):
            if [i] == eq1_label[x]:
                cnts[i]=cnts[i]+1
  
    for z in range(k):    
        for i in range(rows):
            if [z] == eq1_label[i]:
                x2_all[z]=x2_all[z]+df[i]
    
    for i in range(k):
        new_cent[i]=np.absolute(x2_all[i]/cnts[i])
     
    for i in range(k):
        for j in range(cols):
            new_cent[i][j]=round(new_cent[i][j],1)
            
    return new_cent

def new_distance2(df,cen,rows,k,cols):
                
    x2_all=np.zeros(shape=(k,cols))
    new_cent=np.zeros(shape=(k,cols))
    cnts=np.zeros(shape=(k,1))
    x3=[]
    x3_all=np.zeros(shape=(rows,k))
  
    for i in df:
        for j in cen:
            term = 0.0
            termx = 0.0
            termy = 0.0
            for p, l in zip(i, j):
                if (p > l):
                    difference = p - l
                    termx += difference/max(absolute(p),absolute(l),absolute(p-l))
                elif (p < l):
                    difference = l - p
                    termy += difference/max(absolute(p),absolute(l),absolute(l-p))
            termsqr = np.square(termx) + np.square(termy)
            term = np.sqrt(termsqr)
            x3.append(term)
            
    #print x3
    #print '\n'
    #print len(x3)
            
    x3= np.reshape(x3,(rows,k)) 
    
    for i in range(rows):
        for j in range(k):            
            x3_all[i][j]=x3[i][j]

    for i in range(rows): 
        eq2_label[i]=list(x3_all.argsort()[i][:1])
    
    for i in range(k):
        for x in range(rows):
            if [i] == eq2_label[x]:
                cnts[i]=cnts[i]+1
  
    for z in range(k):    
        for i in range(rows):
            if [z] == eq2_label[i]:
                x2_all[z]=x2_all[z]+df[i]
    
    for i in range(k):
        new_cent[i]=np.absolute(x2_all[i]/cnts[i])
     
    for i in range(k):
        for j in range(cols):
            new_cent[i][j]=round(new_cent[i][j],1)
            
    return new_cent

#Main Function

while True:
    dist=input(" Enter:"
               "\n"
               "1 for K means clustering using Eucledian Distance metric"
               "\n"
               "2 for Elkans Accelarated Approach"
               "\n"
               "3 for using Cosine Distance metric" 
               "\n"
               "4 for using Equation 1 as Distance metric"
               "\n"
               "5 for using Equation 2 as Distance metric"
               "\n"
               "6 for using Cityblock as Distance metric"
               "\n"
               "7 to exit")
    
    dfil=input(" Select the dataset to be used:"
               "\n"
               "1 for Clustering based on flower species on Iris Dataset"
               "\n"
               #"2 for Acoustic features extracted from syllables of anuran (frogs) calls to cluster based on family of frog"
               "2 for Clustering based on Wine Data"
               "\n"
               "3 for Clustering of student data based on final exam result" 
               "\n")
    k=input(" Actual K values for Iris data ( K=3 ), Wine Quality Data ( K=3 ),Student data ( K=4 )"
            "\n"
            "Enter the K value -  ")
    
    print('\n')
    
    # The file Pathway has to be editted inorder to be read
    
    #if running on server please choose the appropriate read statement from the options below
    
    if dfil==1:
        dataframe=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
       #dataframe=pd.read_csv("iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
        
    elif dfil==2:
        dataframe=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\wine.csv",header=0)
        #dataframe=pd.read_csv("wine.csv",header=0)
        
    elif dfil==3:
        dataframe=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\student.csv",header=0)
        #dataframe=pd.read_csv("student.csv",header=0)
        
    dataframe1=dataframe.drop('Class Label', 1) # dropping the class label column
    df=dataframe1.values
    l=np.shape(df)
    rows=int(l[0]) # To get the total number of rows in the dataset
    cols=int(l[1]) 
    cen=np.asarray(random.sample(df,k)) # Assigning K Random centroids
    print'\n' 
    print 'Initial Centroid -','\n', cen
    print'\n'  
    
    if dist==1:
        
        all_cen_new=np.zeros(shape=(k,cols))
        
        nu=0
        
        while True:
            sse_euc=np.zeros(shape=(rows,1))
            sse_total_euc=0
            nu=nu+1
            all_cen_new=euclidean_distance(df,cen,rows,k,cols)
            for i in range(rows):
                sse_euc[i]=(np.sum((df[i] - cen[euc_label[i]]) ** 2))       
            if np.array_equal(all_cen_new,cen):
                break;
            print'\n'     
            print'total SSE'        
            print sum(sse_euc)
            print'\n'     
            print'Centroid ',nu,'\n',all_cen_new
            cen=all_cen_new
        
        #print'Cluster Labels','\n' 
        #print euc_label
        
        #Printing the labels in each clusters
    
        dataframe['Cluster'] = [1]*len(dataframe)
        
        for i in range(rows):
            temp = list(euc_label[i])
            temp2 = temp.pop()
            dataframe['Cluster'][i]=temp2
        print'Printing the labels in each clusters''\n'    
        tab = dataframe.groupby(['Cluster', 'Class Label']).size()   
        print tab
        
        # Calculating the RANDOM INDEX inorder to determine the quality of the cluster we found above
        
        # This part of the code is used to calculate the random index inorder to determine the cluster quality

        if dfil==1:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
           #tdf=pd.read_csv("iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
        
        elif dfil==2:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\wine.csv",header=0)
            #tdf=pd.read_csv("wine.csv",header=0)
        
        elif dfil==3:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\student.csv",header=0)
            #tdf=pd.read_csv("student.csv",header=0)
            
            
        if dfil==1:
            columns = ['sepal length','sepal width','petal length','petal width']
            
        elif dfil==2:
            columns = ['alcohol','malic acid','ash','Alcalinity of ash','magnesium','total phenols','flavanoids','Nonflavanoid phenols','Proanthocyanins','color intensity','hue','OD280/OD315 of diluted wines','proline']
            
        elif dfil==3:
            columns = ['gender','region','highest_education','num_of_prev_attempts','studied_credits','disability']

        
        tdff=tdf.drop(columns,1)
        
        tdff['Cluster label'] = [1]*len(dataframe)
        
        for i in range(rows):
            temp=list(euc_label[i])
            temp1=temp.pop()
            tdff['Cluster label'][i]=temp1
        
        tdff['Class label categorical'] = tdff['Class Label'].astype('category')
        tdff['Class label categorical'] = tdff['Class label categorical'].cat.codes
        tdff['Cluster label categorical'] = tdff['Cluster label'].astype('category')
        tdff['Cluster label categorical'] = tdff['Cluster label categorical'].cat.codes

        df1=np.zeros(shape=(rows,4))
        df1=tdff.values
        l1=np.shape(df1)
        rows1=int(l1[0]) # To get the total number of rows in the dataset
        cols1=int(l1[1]) 
        t1=np.zeros(shape=(k,k)) 
        num=0
        den=0
        
        df2=np.split(df1[:, 3], np.cumsum(np.unique(df1[:, 0], return_counts=True)[1])[:-1])
        cls=np.zeros(shape=(k,k))
        for i in range(k):
            for j in range(k):
                cls[i][j]=list(df2[i]).count(j)
        
        clss=map(list, zip(*cls))
        for i in range(k):
            index_max = max(xrange(len(clss[i])), key=clss[i].__getitem__)
            for j in range(k):
                if j!=index_max:
                    den=den+clss[i][j]
                    
        for i in range(k):
            num=num+max(clss[i])
            
        rand_index=num/(den+num)
        
        print '\n',' The True postive + True Negative is- ',num
        print '\n',' The False postive + Flase Negative is - ',den
        print '\n',' The cluster quality measured using Random Index is - ',rand_index
        
        
    elif dist==2:    # Elkans Accelaration
        
        lb_all=np.zeros(shape=(rows,k)) # lower bound
        ub_all=np.zeros(shape=(rows,1)) # upper bound
    
        for i in range(rows): # Initia;izing upper bounds with distance b/w pt x and cluster 1 (default)
            ub_all[i]=np.absolute(np.sqrt(np.sum((df[i] - cen[0]) ** 2)))
            
        for i in range(rows): # a(i) from the algorithm = elk_label[] (dict)
            elk_label[i]=[0]
        
        all_cen_new1=np.zeros(shape=(k,cols))
        
        nu1=0
        
        while True:
            nu1=nu1+1
            
            all_cen_new1=elkans(df,cen,rows,k,cols,lb_all,ub_all,elk_label)
    
            if np.array_equal(all_cen_new1,cen):
                break;
                
            print'\n'     
            print'Centroid ',nu1,'\n',all_cen_new1
            cen=all_cen_new1
                    
        #Printing the labels in each clusters
    
        dataframe['Cluster_elk'] = [1]*len(dataframe)
        
        for i in range(rows):
            temp = list(elk_label[i])
            temp2 = temp.pop()
            dataframe['Cluster_elk'][i]=temp2
        print'Printing the labels in each clusters''\n'    
        tab1 = dataframe.groupby(['Cluster_elk', 'Class Label']).size()   
        print tab1
        
         # Calculating the RANDOM INDEX inorder to determine the quality of the cluster we found above
        
        # This part of the code is used to calculate the random index inorder to determine the cluster quality

        if dfil==1:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
           #tdf=pd.read_csv("iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
        
        elif dfil==2:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\wine.csv",header=0)
            #tdf=pd.read_csv("wine.csv",header=0)
        
        elif dfil==3:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\student.csv",header=0)
            #tdf=pd.read_csv("student.csv",header=0)
        
        if dfil==1:
            columns = ['sepal length','sepal width','petal length','petal width']
            
        elif dfil==2:
            columns = ['alcohol','malic acid','ash','Alcalinity of ash','magnesium','total phenols','flavanoids','Nonflavanoid phenols','Proanthocyanins','color intensity','hue','OD280/OD315 of diluted wines','proline']
            
        elif dfil==3:
            columns = ['gender','region','highest_education','num_of_prev_attempts','studied_credits','disability']
            
        tdff=tdf.drop(columns,1)
        
        tdff['Cluster label'] = [1]*len(dataframe)
        
        for i in range(rows):
            temp=list(elk_label[i])
            temp1=temp.pop()
            tdff['Cluster label'][i]=temp1
        
        tdff['Class label categorical'] = tdff['Class Label'].astype('category')
        tdff['Class label categorical'] = tdff['Class label categorical'].cat.codes
        tdff['Cluster label categorical'] = tdff['Cluster label'].astype('category')
        tdff['Cluster label categorical'] = tdff['Cluster label categorical'].cat.codes

        df1=np.zeros(shape=(rows,4))
        df1=tdff.values
        l1=np.shape(df1)
        rows1=int(l1[0]) # To get the total number of rows in the dataset
        cols1=int(l1[1]) 
        t1=np.zeros(shape=(k,k)) 
        num=0
        den=0
        
        df2=np.split(df1[:, 3], np.cumsum(np.unique(df1[:, 0], return_counts=True)[1])[:-1])
        cls=np.zeros(shape=(k,k))
        for i in range(k):
            for j in range(k):
                cls[i][j]=list(df2[i]).count(j)
        
        clss=map(list, zip(*cls))
        for i in range(k):
            index_max = max(xrange(len(clss[i])), key=clss[i].__getitem__)
            for j in range(k):
                if j!=index_max:
                    den=den+clss[i][j]
                    
        for i in range(k):
            num=num+max(clss[i])
            
        rand_index=num/(den+num)
        
        print '\n',' The True postive + True Negative is- ',num
        print '\n',' The False postive + Flase Negative is - ',den
        print '\n',' The cluster quality measured using Random Index is - ',rand_index

    elif dist==3:
        
        all_cen_new=np.zeros(shape=(k,cols))
        
        nu2=0
        
        while True:
            sse_cos=np.zeros(shape=(rows,1))
            sse_total_cos=0
            nu2=nu2+1
            all_cen_new=cosine_distance(df,cen,rows,k,cols)
            for i in range(rows):
                sse_cos[i]=(np.sum((df[i] - cen[cos_label[i]]) ** 2))       
            if np.array_equal(all_cen_new,cen):
                break;
            print'\n'     
            print'total SSE'        
            print sum(sse_cos)
            print'\n'     
            print'Centroid ',nu2,'\n',all_cen_new
            cen=all_cen_new
        
        #print'Cluster Labels','\n' 
        #print cos_label
              
        #Printing the labels in each clusters
    
        dataframe['Cluster'] = [1]*len(dataframe)
        
        for i in range(rows):
            temp = list(cos_label[i])
            temp2 = temp.pop()
            dataframe['Cluster'][i]=temp2
        print'Printing the labels in each clusters''\n'    
        tab3 = dataframe.groupby(['Cluster', 'Class Label']).size()   
        print tab3
        
         # Calculating the RANDOM INDEX inorder to determine the quality of the cluster we found above
        
        # This part of the code is used to calculate the random index inorder to determine the cluster quality
        
        if dfil==1:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
           #tdf=pd.read_csv("iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
        
        elif dfil==2:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\wine.csv",header=0)
            #tdf=pd.read_csv("wine.csv",header=0)
        
        elif dfil==3:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\student.csv",header=0)
            #tdf=pd.read_csv("student.csv",header=0)
        
        if dfil==1:
            columns = ['sepal length','sepal width','petal length','petal width']
            
        elif dfil==2:
            columns = ['alcohol','malic acid','ash','Alcalinity of ash','magnesium','total phenols','flavanoids','Nonflavanoid phenols','Proanthocyanins','color intensity','hue','OD280/OD315 of diluted wines','proline']
            
        elif dfil==3:
            columns = ['gender','region','highest_education','num_of_prev_attempts','studied_credits','disability']

        tdff=tdf.drop(columns,1)
        
        tdff['Cluster label'] = [1]*len(dataframe)
        
        for i in range(rows):
            temp=list(cos_label[i])
            temp1=temp.pop()
            tdff['Cluster label'][i]=temp1
        
        tdff['Class label categorical'] = tdff['Class Label'].astype('category')
        tdff['Class label categorical'] = tdff['Class label categorical'].cat.codes
        tdff['Cluster label categorical'] = tdff['Cluster label'].astype('category')
        tdff['Cluster label categorical'] = tdff['Cluster label categorical'].cat.codes

        df1=np.zeros(shape=(rows,4))
        df1=tdff.values
        l1=np.shape(df1)
        rows1=int(l1[0]) # To get the total number of rows in the dataset
        cols1=int(l1[1]) 
        t1=np.zeros(shape=(k,k)) 
        num=0
        den=0
        
        df2=np.split(df1[:, 3], np.cumsum(np.unique(df1[:, 0], return_counts=True)[1])[:-1])
        cls=np.zeros(shape=(k,k))
        for i in range(k):
            for j in range(k):
                cls[i][j]=list(df2[i]).count(j)
        
        clss=map(list, zip(*cls))
        for i in range(k):
            index_max = max(xrange(len(clss[i])), key=clss[i].__getitem__)
            for j in range(k):
                if j!=index_max:
                    den=den+clss[i][j]
                    
        for i in range(k):
            num=num+max(clss[i])
            
        rand_index=num/(den+num)
        
        print '\n',' The True postive + True Negative is- ',num
        print '\n',' The False postive + Flase Negative is - ',den
        print '\n',' The cluster quality measured using Random Index is - ',rand_index
        
    elif dist==4:
        all_cen_new=np.zeros(shape=(k,cols))
        
        nu3=0
        
        while True:
            sse_eq1=np.zeros(shape=(rows,1))
            sse_total_eq1=0
            nu3=nu3+1
            all_cen_new=new_distance1(df,cen,rows,k,cols)
            for i in range(rows):
                sse_eq1[i]=(np.sum((df[i] - cen[eq1_label[i]]) ** 2))       
            if np.array_equal(all_cen_new,cen):
                break;
            print'\n'     
            print'total SSE'        
            print sum(sse_eq1)
            print'\n'     
            print'Centroid ',nu3,'\n',all_cen_new
            cen=all_cen_new
        
        #print'Cluster Labels','\n' 
        #print eq1_label
              
        #Printing the labels in each clusters
    
        dataframe['Cluster'] = [1]*len(dataframe)
        
        for i in range(rows):
            temp = list(eq1_label[i])
            temp2 = temp.pop()
            dataframe['Cluster'][i]=temp2
        print'Printing the labels in each clusters''\n'    
        tab4 = dataframe.groupby(['Cluster', 'Class Label']).size()   
        print tab4 
        
        # This part of the code is used to calculate the random index inorder to determine the cluster quality
        if dfil==1:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
           #tdf=pd.read_csv("iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
        
        elif dfil==2:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\wine.csv",header=0)
            #tdf=pd.read_csv("wine.csv",header=0)
        
        elif dfil==3:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\student.csv",header=0)
            #tdf=pd.read_csv("student.csv",header=0)
        
        if dfil==1:
            columns = ['sepal length','sepal width','petal length','petal width']
            
        elif dfil==2:
            columns = ['alcohol','malic acid','ash','Alcalinity of ash','magnesium','total phenols','flavanoids','Nonflavanoid phenols','Proanthocyanins','color intensity','hue','OD280/OD315 of diluted wines','proline']
            
        elif dfil==3:
            columns = ['gender','region','highest_education','num_of_prev_attempts','studied_credits','disability']
            
        tdff=tdf.drop(columns,1)
        
        tdff['Cluster label'] = [1]*len(dataframe)
        
        for i in range(rows):
            temp=list(eq1_label[i])
            temp1=temp.pop()
            tdff['Cluster label'][i]=temp1
        
        tdff['Class label categorical'] = tdff['Class Label'].astype('category')
        tdff['Class label categorical'] = tdff['Class label categorical'].cat.codes
        tdff['Cluster label categorical'] = tdff['Cluster label'].astype('category')
        tdff['Cluster label categorical'] = tdff['Cluster label categorical'].cat.codes

        df1=np.zeros(shape=(rows,4))
        df1=tdff.values
        l1=np.shape(df1)
        rows1=int(l1[0]) # To get the total number of rows in the dataset
        cols1=int(l1[1]) 
        t1=np.zeros(shape=(k,k)) 
        num=0
        den=0
        
        df2=np.split(df1[:, 3], np.cumsum(np.unique(df1[:, 0], return_counts=True)[1])[:-1])
        cls=np.zeros(shape=(k,k))
        for i in range(k):
            for j in range(k):
                cls[i][j]=list(df2[i]).count(j)
        
        clss=map(list, zip(*cls))
        for i in range(k):
            index_max = max(xrange(len(clss[i])), key=clss[i].__getitem__)
            for j in range(k):
                if j!=index_max:
                    den=den+clss[i][j]
                    
        for i in range(k):
            num=num+max(clss[i])
            
        rand_index=num/(den+num)
        
        print '\n',' The True postive + True Negative is- ',num
        print '\n',' The False postive + Flase Negative is - ',den
        print '\n',' The cluster quality measured using Random Index is - ',rand_index
        
    
    elif dist==5:
        all_cen_new=np.zeros(shape=(k,cols))
        
        nu4=0
        
        while True:
            sse_eq2=np.zeros(shape=(rows,1))
            sse_total_eq2=0
            nu4=nu4+1
            all_cen_new=new_distance2(df,cen,rows,k,cols)
            for i in range(rows):
                sse_eq2[i]=(np.sum((df[i] - cen[eq2_label[i]]) ** 2))       
            if np.array_equal(all_cen_new,cen):
                break;
            print'\n'     
            print'total SSE'        
            print sum(sse_eq2)
            print'\n'     
            print'Centroid ',nu4,'\n',all_cen_new
            cen=all_cen_new
        
        #print'Cluster Labels','\n' 
        #print eq2_label
              
        #Printing the labels in each clusters
    
        dataframe['Cluster'] = [1]*len(dataframe)
        
        for i in range(rows):
            temp = list(eq2_label[i])
            temp2 = temp.pop()
            dataframe['Cluster'][i]=temp2
        print'Printing the labels in each clusters''\n'    
        tab5 = dataframe.groupby(['Cluster', 'Class Label']).size()   
        print tab5 
        
        # This part of the code is used to calculate the random index inorder to determine the cluster quality

        if dfil==1:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
           #tdf=pd.read_csv("iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
        
        elif dfil==2:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\wine.csv",header=0)
            #tdf=pd.read_csv("wine.csv",header=0)
        
        elif dfil==3:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\student.csv",header=0)
            #tdf=pd.read_csv("student.csv",header=0)
        
        if dfil==1:
            columns = ['sepal length','sepal width','petal length','petal width']
            
        elif dfil==2:
            columns = ['alcohol','malic acid','ash','Alcalinity of ash','magnesium','total phenols','flavanoids','Nonflavanoid phenols','Proanthocyanins','color intensity','hue','OD280/OD315 of diluted wines','proline']
            
        elif dfil==3:
            columns = ['gender','region','highest_education','num_of_prev_attempts','studied_credits','disability']
            
        tdff=tdf.drop(columns,1)
        
        tdff['Cluster label'] = [1]*len(dataframe)
        
        for i in range(rows):
            temp=list(eq2_label[i])
            temp1=temp.pop()
            tdff['Cluster label'][i]=temp1
        
        tdff['Class label categorical'] = tdff['Class Label'].astype('category')
        tdff['Class label categorical'] = tdff['Class label categorical'].cat.codes
        tdff['Cluster label categorical'] = tdff['Cluster label'].astype('category')
        tdff['Cluster label categorical'] = tdff['Cluster label categorical'].cat.codes

        df1=np.zeros(shape=(rows,4))
        df1=tdff.values
        l1=np.shape(df1)
        rows1=int(l1[0]) # To get the total number of rows in the dataset
        cols1=int(l1[1]) 
        t1=np.zeros(shape=(k,k)) 
        num=0
        den=0
        
        df2=np.split(df1[:, 3], np.cumsum(np.unique(df1[:, 0], return_counts=True)[1])[:-1])
        cls=np.zeros(shape=(k,k))
        for i in range(k):
            for j in range(k):
                cls[i][j]=list(df2[i]).count(j)
        
        clss=map(list, zip(*cls))
        for i in range(k):
            index_max = max(xrange(len(clss[i])), key=clss[i].__getitem__)
            for j in range(k):
                if j!=index_max:
                    den=den+clss[i][j]
                    
        for i in range(k):
            num=num+max(clss[i])
            
        rand_index=num/(den+num)
        
        print '\n',' The True postive + True Negative is- ',num
        print '\n',' The False postive + Flase Negative is - ',den
        print '\n',' The cluster quality measured using Random Index is - ',rand_index
        
    elif dist==6:
                
        all_cen_new=np.zeros(shape=(k,cols))
        
        nu6=0
        
        while True:
            sse_city=np.zeros(shape=(rows,1))
            sse_total_city=0
            nu6=nu6+1
            all_cen_new=cityblock_distance(df,cen,rows,k,cols)
            for i in range(rows):
                sse_city[i]=(np.sum((df[i] - cen[city_label[i]]) ** 2))       
            if np.array_equal(all_cen_new,cen):
                break;
            print'\n'     
            print'total SSE'        
            print sum(sse_city)
            print'\n'     
            print'Centroid ',nu6,'\n',all_cen_new
            cen=all_cen_new
        
        #print'Cluster Labels','\n' 
        #print euc_label
        
        #Printing the labels in each clusters
    
        dataframe['Cluster'] = [1]*len(dataframe)
        
        for i in range(rows):
            temp = list(city_label[i])
            temp2 = temp.pop()
            dataframe['Cluster'][i]=temp2
        print'Printing the labels in each clusters''\n'    
        tab = dataframe.groupby(['Cluster', 'Class Label']).size()   
        print tab
        
        # Calculating the RANDOM INDEX inorder to determine the quality of the cluster we found above
        
        # This part of the code is used to calculate the random index inorder to determine the cluster quality

        if dfil==1:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
           #tdf=pd.read_csv("iris.csv",header=None,names=["sepal length","sepal width","petal length","petal width","Class Label"])
        
        elif dfil==2:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\wine.csv",header=0)
            #tdf=pd.read_csv("wine.csv",header=0)
        
        elif dfil==3:
            tdf=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\student.csv",header=0)
            #tdf=pd.read_csv("student.csv",header=0)
            
            
        if dfil==1:
            columns = ['sepal length','sepal width','petal length','petal width']
            
        elif dfil==2:
            columns = ['alcohol','malic acid','ash','Alcalinity of ash','magnesium','total phenols','flavanoids','Nonflavanoid phenols','Proanthocyanins','color intensity','hue','OD280/OD315 of diluted wines','proline']
            
        elif dfil==3:
            columns = ['gender','region','highest_education','num_of_prev_attempts','studied_credits','disability']

        
        tdff=tdf.drop(columns,1)
        
        tdff['Cluster label'] = [1]*len(dataframe)
        
        for i in range(rows):
            temp=list(city_label[i])
            temp1=temp.pop()
            tdff['Cluster label'][i]=temp1
        
        tdff['Class label categorical'] = tdff['Class Label'].astype('category')
        tdff['Class label categorical'] = tdff['Class label categorical'].cat.codes
        tdff['Cluster label categorical'] = tdff['Cluster label'].astype('category')
        tdff['Cluster label categorical'] = tdff['Cluster label categorical'].cat.codes

        df1=np.zeros(shape=(rows,4))
        df1=tdff.values
        l1=np.shape(df1)
        rows1=int(l1[0]) # To get the total number of rows in the dataset
        cols1=int(l1[1]) 
        t1=np.zeros(shape=(k,k)) 
        num=0
        den=0
        
        df2=np.split(df1[:, 3], np.cumsum(np.unique(df1[:, 0], return_counts=True)[1])[:-1])
        cls=np.zeros(shape=(k,k))
        for i in range(k):
            for j in range(k):
                cls[i][j]=list(df2[i]).count(j)
        
        clss=map(list, zip(*cls))
        for i in range(k):
            index_max = max(xrange(len(clss[i])), key=clss[i].__getitem__)
            for j in range(k):
                if j!=index_max:
                    den=den+clss[i][j]
                    
        for i in range(k):
            num=num+max(clss[i])
            
        rand_index=num/(den+num)
        
        print '\n',' The True postive + True Negative is- ',num
        print '\n',' The False postive + Flase Negative is - ',den
        print '\n',' The cluster quality measured using Random Index is - ',rand_index
        
    elif dist==7:
        break;
       
        
    


# In[ ]:



