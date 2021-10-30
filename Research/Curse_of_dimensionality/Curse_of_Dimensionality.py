
# coding: utf-8

# In[5]:

import numpy as np
import random
import math as m
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

n=100 # Set Number of data points as per requirement 
k=101  # Set Number of dimensions as per requirement

#Dictionary consisting of all R(k) values with keys as the dimension

#r1 has Euclidean Distance R(k) values
#r2 has Citiblock Distance R(k) values
#r3 has Minkowski Distance R(k) values
#r4 has Cosine Distance R(k) values
#r5 has Problem 4 Distance Function R(k) values

r1_all={}
r2_all={}
r3_all={}
r4_all={}
r5_all={}

#Using a global counter variable for iterating the dictionary

global count
count=1

# Formulating the different Distance Functions to be evaluated

def euclidean_distance(m):
   
    x1=[]
    for i in range(n):
        for j in range(n):
            if not np.array_equal(m[i,:],m[j,:]):  
                x1.append(np.absolute(np.sqrt(np.sum((m[i,:]-m[j,:])**2))))
    #print("input : ")
    #print(m)
    
    #print("euclidean distance is : ")
    #print(x1)
    
    r1=np.log10((np.max(x1)-np.min(x1))/np.min(x1))
    #print("R value is : ",r1)
    
    r1_all[count]=r1
    
    return x1

def cityblock_distance(m):
    x2=[]
    for i in range(n):
        for j in range(n):
            if not np.array_equal(m[i,:],m[j,:]):  
                x2.append(np.sum((np.absolute((m[i,:]-m[j,:])))))
    #print("input : ")
    #print(m)
    
    #print("cityblok distance is : ")
    #print(x1)
    
    r2=np.log10((np.max(x2)-np.min(x2))/np.min(x2))
    #print("R value is : ",r2)
    
    r2_all[count]=r2
    
    
    return x2

def minkowski_distance(m):

    x3=[]
    for i in range(n):
        for j in range(n):
            if not np.array_equal(m[i,:],m[j,:]):            
                #x3.append(np.power(np.sum(np.power(np.absolute((m[i,:]-m[j,:])),3)),1/3))
                x3.append(np.absolute(np.cbrt(np.sum((m[i,:]-m[j,:])**3))))
       
    #print("input : ")
    #print(m)
    
    #print("Minkowski distance is : ")
    #print(x3)
    
    r3=np.log10((np.max(x3)-np.min(x3))/np.min(x3))
    #print("R value is : ",r3)
    
    r3_all[count]=r3
    
    return x3

def cosine_distance(m):
  
    x4=[]
    for i in range(n):
        for j in range(i+1,n):
            if not np.array_equal(m[i,:],m[j,:]):
                val = 1- (np.dot(m[i,:],m[j,:])/(np.linalg.norm(m[i,:])*np.linalg.norm(m[j,:])))
                x4.append(val) 
    #print("input : ")
    #print(m)
    
    #print("cosine distance is : ")
    #print(x4)
    
    r4=np.log10((np.max(x4)-np.min(x4))/np.min(x4))
    #print("R value is : ",r1)
    
    r4_all[count]=r4
    
    
    return x4

def fourth_distance(m):

    x5=[]
    y1=[]
    y2=[]
    for i in m:
        for j in m:
            if not np.array_equal(i,j):
                term=0.0
                termx=0.0
                termy=0.0
                for k,l in zip(i,j):
                    if(k>l):
                        difference=k-l
                        termx += difference
                    elif(k<l):
                        difference=l-k
                        termy += difference
                termsqr=np.square(termx)+np.square(termy)
                term=np.sqrt(termsqr)
                x5.append(term)
    #print("input : ")
    #print(m)
    
    #print("euclidean distance is : ")
    #print(x5)
    

    #x5.append(np.absolute(np.sqrt(np.sum(y1,y2))))   

    r5=np.log10((np.max(x5)-np.min(x5))/np.min(x5))
    #print("R value is : ",r5)
    
    r5_all[count]=r5
    
    
    return x5

# Main Function

for d in range(2,k):
    
    #Implemented it for uniform and gaussian random number generator with a zero-mean unit-covariance matrix 
    
    #Choose the appropriate random number generator
    
    #m=np.random.random((n,d)) #uniform random number generator
    m=np.random.normal(loc=0, scale=1, size=(n,d))  # Gaussian random number Generator
    
    #Iterating over all the distance functions for all dimensions upto 100
    euclidean_distance(m)
    cityblock_distance(m)
    minkowski_distance(m)
    cosine_distance(m)
    fourth_distance(m)
    count=count+1
        #print("K Value is (Dimension) : ",d) 
        
print("The values of r1 are : ")        
print(r1_all)     
print("The values of r2 are : ") 
print(r2_all)
print("The values of r3 are : ") 
print(r3_all)    
print("The values of r4 are : ") 
print(r4_all) 
print("The values of r5 are : ") 
print(r5_all) 

# Using matplotlib to plot R(k) as a function of k

fig = plt.figure()

g = fig.add_axes([0,0,1,1])

blue_patch = mpatches.Patch(color='blue', label='Euclidean Distance')
red_patch = mpatches.Patch(color='red', label='Citiblock Distance')
yellow_patch = mpatches.Patch(color='yellow', label='Minkowski Distance')
green_patch = mpatches.Patch(color='green', label='Cosine Distance')
black_patch = mpatches.Patch(color='black', label='Problem 4 Distance Function')

g.legend(handles=[blue_patch,red_patch,yellow_patch,green_patch,black_patch])
g.plot(r1_all.keys(),r1_all.values(),color = 'blue',alpha = 0.7)
g.plot(r2_all.keys(),r2_all.values(),color = 'red',alpha = 0.7)
g.plot(r3_all.keys(),r3_all.values(),color = 'yellow',alpha = 0.7)
g.plot(r4_all.keys(),r4_all.values(),color = 'green',alpha = 0.7)
g.plot(r5_all.keys(),r5_all.values(),color = 'black',alpha = 0.7)

# change the labels accordingly

#g.set_title('The Curse Of Dimensionality for n=100 with Uniform random generator between 0 and 1')
g.set_title('The Curse Of Dimensionality for n=100 with a zero-mean unit-covariance matrix Gaussian source')

g.set_xlabel(' Dimensions (k)')
g.set_ylabel(' R(k) ')

fig


# In[ ]:



