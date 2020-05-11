#Loading the data into R 

library("ggplot2")
data = read.csv(file.choose(),header = F)
d1 = data

# Initializing

ggplot(d1,aes(sample = d1$V1)) + stat_qq()
gaussian =3
m <- vector()
N_m <- vector()
sd <- vector()
N_sd <- vector()
w <- vector()
N_w <- vector()
Prob_Xg <- matrix(0, nrow = nrow(d1), ncol = gaussian)
PostProb_gX <- matrix(0, nrow = nrow(d1), ncol = gaussian)

# Initializing the GAUSSIAN PARAMETERS 

# Initializing mean
for(k in 1:gaussian){
    m[k]= k * 1/gaussian * (max(d1) - min(d1))
}
sd<-runif(gaussian,0,1)

# Initializing weights for gaussians
for(k in 1:gaussian){
    w[k]= 1/gaussian
}

# RUNNING AN ITERATION ON L 
L <- vector()
Counter = 0
while(Counter == 0){
# Expectation
# Probability of point X(i)|g(k)
    for(i in 1:nrow(d1)){
        for(k in 1:gaussian){
            Prob_Xg[i,k]= ( 1/(sqrt(2*pi*sd[k]^2)) * exp(-1 * ((d1[i,1]-m[k])^2) / (2* sd[k]^2)))
            }
    }
    
# Posterior Probability: g(i)|X
    for(i in 1:nrow(d1)){
        for(k in 1:gaussian){
            PostProb_gX[i,k]= (Prob_Xg[i,k] * w[k]) / (sum(Prob_Xg[i,] * w))
        }
    }
    N_i<-vector()
  for(k in 1:gaussian){
    N_i[k] = sum(PostProb_gX[,k])}
    
# Maximization 
    
    for(k in 1:gaussian){
        
# Assigning new weights
        N_w[k] = N_i[k]/sum(N_i)
        
# Assigning new Mean
        N_m[k] = sum(PostProb_gX[,k] * d1)/ sum(PostProb_gX[,k])
        
# Assigning new Variance
        N_sd[k] = sqrt(sum(PostProb_gX[,k] * (d1 - N_m[k])^2) / sum(PostProb_gX[,k]))
    }
    
    N = sum(N_w)
    w = N_w
    m = N_m 
    sd = N_sd 
    
# Log-likelihood
    
    L_hat = 0
    for (j in 1:nrow(d1))
    {
      temp<-0
      for(k in 1:gaussian)
        temp = temp + (Prob_Xg[j,k] * N_w[k])
      L_hat = sum(L_hat, log(temp))
    }
    
    L = append(L, L_hat)
    
# Loop to stop iteration when we reach maximum likelihood
    
    if (length(abs(L[length(L)] - L[length(L)-1]))==0) {
        Counter = 0
    }
    else if (abs(L[length(L)] - L[length(L)-1]) > 0.00001) {
        Counter = 0
    }
    else
        Counter = 1    

}

plot(2:length(L),L[-1], main = "Log Likelihood across iterations", xlab = "Iteration",
     ylab = "Log Likelihood")

# Printing the values :
# m
# sd
# w
# max(L)