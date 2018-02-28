
# coding: utf-8

# In[6]:

import numpy as np
import random
import numpy as np
import random
import math as m
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

global dim
dim=3  # Please change the dimension value as per requirement
global n
n=10000  # N value can be set differently as well

n1dict={}
n2dict={}
n3dict={}
n4dict={}

def euclidean_distance(m):
    k=5
    c, r = n, k+1;
    x1_all=np.zeros(shape=(n,n))

    x1_all1 = [[0 for x in range(r)] for y in range(c)] 

    n1_all=[n]

    n1=[n]

    for i in range(n):
        for j in range(n):
            if not np.array_equal(m[i, :],m[j, :]):
                x1_all[i][j]=np.absolute(np.sqrt(np.sum((m[i, :] - m[j, :]) ** 2)))
            
    for i in range(n): 
        #print(list(x1_all.argsort()[i][:k])) 
        x1_all1[i][:]=list(x1_all.argsort()[i][:k])
     
    #print("x1_all1")                
    #print(x1_all1)  
    
    for x in range(n):
        c=0
        for y in range(n):
            for z in range(k): 
                if x==x1_all1[y][z]:
                    c=c+1
                    #print(x,c)
        n1_all.append(c)
    
    for x in range(n):
        n1.append(n1_all[x+1]-1)
    
    n1.pop(0)
    
    for i in range(n):
        n1dict[i]=n1[i]
    
    print("n1_dict")      
    print(n1dict) 
    
    #print("n1_all")      
    #print(n1_all)   

    #print("Values of N1 are :")  
    #print(n1)
    
    return 0

def cosine_distance(m):
    
    k=5
    
    c, r = n, k+1;
    
    x2_all=np.zeros(shape=(n,n))

    x2_all1 = [[0 for x in range(r)] for y in range(c)] 

    n2_all=[n]

    n2=[n]
    for i in range(n):
        #for j in range(i + 1, n):
        for j in range(n):
            if not np.array_equal(m[i, :], m[j, :]):
                x2_all[i][j] = (1 - (np.dot(m[i, :], m[j, :]) / (np.linalg.norm(m[i, :]) * np.linalg.norm(m[j, :]))))
              
    for i in range(n): 
        #print(list(x2_all.argsort()[i][:k])) 
        x2_all1[i][:]=list(x2_all.argsort()[i][:k])
        
    #print("x2_all1")                
    #print(x2_all1)  
    
    for x in range(n):
        c=0
        for y in range(n):
            for z in range(k): 
                if x==x2_all1[y][z]:
                    c=c+1
                    #print(x,c)
        n2_all.append(c)
    
    for x in range(n):
        n2.append(n2_all[x+1]-1)
    
    n2.pop(0)
        
    for i in range(n):
        n2dict[i]=n2[i]
    
       
    n2dict[0]=0
    
    for i in range(len(n2dict)):
        if n2dict[i]==-1:
            n2dict[i]=0
    
    print("n2_dict")      
    print(n2dict) 
    
    #print("n2_all")      
    #print(n2_all)  
    
    #print("Values of N2 are :")  
    #print(n2)   
    return 0


def new_distance1(m):
    
    k=5  
    c, r = n, k+1;
    x3=[]
    x3_all=np.zeros(shape=(n,n-1))
    x3_all1 = [[0 for x in range(r)] for y in range(c)] 
    n3_all=[n]
    n3=[n]
    
 
    for i in m:
        for j in m:
            if not np.array_equal(i, j):
                term = 0.0
                termx = 0.0
                termy = 0.0
                for k, l in zip(i, j):
                    if (k > l):
                        difference = k - l
                        termx += difference
                    elif (k < l):
                        difference = l - k
                        termy += difference
                termsqr = np.square(termx) + np.square(termy)
                term = np.sqrt(termsqr)
                x3.append(term)
#    print("x3 is ")
#    print(x3)
#    print(type(x3))
#    print(len(x3))
    
    x3= np.reshape(x3, (n,n-1)) 
    #print(x3)
    #print(type(x3))
    #print(len(x3))
    
    for i in range(n):
        for j in range(n-1):            
            x3_all[i][j]=x3[i][j]

    #print("x3_all")        
    #print(x3_all)
    #print(type(x3_all))
    
    for i in range(n): 
        #print(x3_all.argsort()[i][:5]) 
        x3_all1[i][:]=list(x3_all.argsort()[i][:5])
        
    #print("x3_all1")                
    #print(x3_all1)  
    
    for x in range(n):
        c=0
        for y in range(n):
            for z in range(5): 
                if x==x3_all1[y][z]:
                    c=c+1
                    #print(x,c)
        n3_all.append(c)
    
    for x in range(n):
        n3.append(n3_all[x+1]-1)
    
    n3.pop(0)

    for i in range(n):
        n3dict[i]=n3[i]
    
    print("n3_dict")      
    print(n3dict) 
    
    #print("n3_all")      
    #print(n3_all) 

    #print("Values of N3 are :")   
    #print(n3)
    
    return 0

def new_distance2(m):
    
    k=5  
    c, r = n, k+1;
    x4=[]
    x4_all=np.zeros(shape=(n,n-1))
    x4_all1 = [[0 for x in range(r)] for y in range(c)] 
    n4_all=[n]
    n4=[n]

    for i in m:
        for j in m:
            if not np.array_equal(i, j):
                term = 0.0
                termx = 0.0
                termy = 0.0
                for k, l in zip(i, j):
                    if (k > l):
                        difference = k - l
                        termx += difference/max(np.absolute(k),np.absolute(l),np.absolute(k-l))
                    elif (k < l):
                        difference = l - k
                        termy += difference/max(np.absolute(k),np.absolute(l),np.absolute(l-k))
                termsqr = np.square(termx) + np.square(termy)
                term = np.sqrt(termsqr)
                x4.append(term)

    x4= np.reshape(x4, (n,n-1))
    #print(x4)
    #print(type(x4))
    #print(len(x4))
    
    for i in range(n):
        for j in range(n-1):            
            x4_all[i][j]=x4[i][j]

    #print("x4_all")        
    #print(x4_all)
    #print(type(x4_all))
    
    for i in range(n): 
        #print(x4_all.argsort()[i][:5]) 
        x4_all1[i][:]=list(x4_all.argsort()[i][:5])
        
    #print("x4_all1")                
    #print(x4_all1)  
    
    for x in range(n):
        c=0
        for y in range(n):
            for z in range(5): 
                if x==x4_all1[y][z]:
                    c=c+1
                    #print(x,c)
        n4_all.append(c)
    
    for x in range(n):
        n4.append(n4_all[x+1]-1)
    
    n4.pop(0)
    
    for i in range(n):
        n4dict[i]=n4[i]
    
    print("n4_dict")      
    print(n4dict)     
    
    #print("n4_all")      
    #print(n4_all) 

    #print("Values of N4 are :")   
    #print(n4)
    
    return 0


#Main Function

rno_gen=input("Enter 1 for Uniform Random Number Generator or 2 for gaussian random number generator")

if rno_gen==1: 	# Implemented it for uniform random number generator 

    for d in range(1, dim+1):
        m=np.random.random((n,d)) #uniform random number generator

elif rno_gen==2:  # Implemented it for gaussian random number generator with a zero-mean unit-covariance matrix

    for d in range(1, dim+1):
        m = np.random.normal(loc=0, scale=1,size=(n, d))  # Gaussian random number Generator

euclidean_distance(m)
cosine_distance(m)
new_distance1(m)
new_distance2(m)


# Using matplotlib to plot R(k) as a function of k

n11={}
n12={}
n13={}
n14={}
n11=Counter(n1dict.values())
n12=Counter(n2dict.values())
n13=Counter(n3dict.values())
n14=Counter(n4dict.values())

fig = plt.figure()

g = fig.add_axes([0, 0, 1, 1])

blue_patch = mpatches.Patch(color='blue', label='Euclidean Distance')
green_patch = mpatches.Patch(color='green', label='Cosine Distance')
yellow_patch = mpatches.Patch(color='yellow', label='New Distance function 1')
red_patch = mpatches.Patch(color='red', label='New Distance function 2')

g.legend(handles=[blue_patch,green_patch,yellow_patch,red_patch])
g.plot(n11.keys(),n11.values(),color='blue', alpha=0.7)
g.plot(n12.keys(),n12.values(),color='green', alpha=0.7)
g.plot(n13.keys(),n13.values(),color='yellow', alpha=0.7)
g.plot(n14.keys(),n14.values(),color='red', alpha=0.7)

# change the labels accordingly

#use while running on uniform random source, change k value in label

g.set_title('Outliers and Hubs for n=10000 and Dimension k=3 with a uniform random source')

#use while running on gaussian random source, change k value in label

#g.set_title('Outliers and Hubs for n=10000 and Dimension k=3 with a Gaussian random source')

g.set_xlabel('N')
g.set_ylabel('')
plt.show(fig)
fig



# In[ ]:

#gaussian 3000 running

