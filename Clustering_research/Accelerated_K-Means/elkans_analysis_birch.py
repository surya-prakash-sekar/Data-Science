
# coding: utf-8

# In[1]:

from __future__ import division
from numpy import *
import pandas as pd
import numpy as np
import random
import math as m
import collections
import sys 
sys.setrecursionlimit(2000)
elk_label={}

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
                
                ub_all[i]=round(np.absolute(np.sqrt(np.sum((df[i,:]-cen[list(elk_label[i]).pop(),:])** 2))),1) 
                
                r=0; # r is set to false
                    
                if ub_all[i]<=z:
                    continue          
            lb_all[i][j]=round(np.absolute(np.sqrt(np.sum((df[i,:]-cen[j,:])** 2))),1) #abs(df[i][j]-cen[j])
                    
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

#Main Function

while True:
    
    dist=input(" Press 1 - to run the analysis"
            "\n"
            "Press 2 - to exit")
    
    k=input(" Vary between K=3 , K=20 and K=100"
            "\n"
            "Enter the K value -  ")
    
    print('\n')
    
    # The file Pathway has to be editted inorder to be read
    
    #if running on server please choose the appropriate read statement from the options below
 
     
    dataframe=pd.read_csv("C:\Users\Surya\Desktop\IU\Spring 18\Data Mining\HomeWork2\Problem 3\Birch.csv",header=0)
    #dataframe=pd.read_csv("birch.csv",header=0)
    
    df=dataframe.values
    l=np.shape(df)
    rows=int(l[0]) # To get the total number of rows in the dataset
    cols=int(l[1]) 
    cen=np.asarray(random.sample(df,k)) # Assigning K Random centroids
    print'\n' 
    print 'Initial Centroid -','\n', cen
    print'\n'          
        
    if dist==1:    # Elkans Accelaration
        
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
        
        print ' The total number of iterations is : ',nu1   
        
    elif dist==2:
        break;

