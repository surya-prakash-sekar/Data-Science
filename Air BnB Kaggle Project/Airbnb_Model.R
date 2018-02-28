#Libraries
install.packages("c50")
library(C50)			 # decision tree
install.packages("lubridate")
library(lubridate) # date functions
install.packages("dplyr")
library(dplyr)		 # data selection and filter
install.packages("caret")
library(caret)		 # creating data partition
library(randomForest) # Random forest

#Loading the data set
getwd()
setwd("C:\\Users\\Surya\\Desktop\\IU\\Fall 17\\Applied Data Science\\Assignment\\Final Project")
train_data<-read.csv("train_users_2.csv",header=TRUE)
test_data<-read.csv("test_users.csv",header=TRUE)

#Data Cleaning

#check for null values columnwise
apply(apply(train_data,2,is.na),2,sum)
apply(apply(test_data,2,is.na),2,sum)

# creating arbitrary Ids in the two datasets
train_data$IDA<-1:nrow(train_data)			# Create an arbitrary index for separation
test_data$IDA<-300000 : 300000 + nrow(test_data)

#  We need to bind the train and test data together, process the data and then separate
# From he training data we need to tremove the variable 'country_destination' and store it somewhere else

labels<-train_data[,c(1,16)]			# Extracted the country_destination column in the training data
train_data$country_destination<-NULL	# removing 'countrty_destination' column in the training data before binding
all_data<-rbind(train_data,test_data)		# Binding train and test data
all_data$date_first_booking<-NULL		# Removing date_first_booking col

# Convert date_account_created to date and then extract individual components year, month, day and day of week
all_data$date_account_created<-ymd(all_data$date_account_created)
all_data$account_created_year<-year(all_data$date_account_created)
all_data$account_created_day<-day(all_data$date_account_created)
all_data$account_created_month<-month(all_data$date_account_created)
all_data$account_created_wday<-wday(all_data$date_account_created)
all_data$date_account_created<-NULL

# Applying above technique for 'timestamp_first_active'
all_data$timestamp_first_active<-ymd_hms(all_data$timestamp_first_active)
all_data$ts_first_active_year<-year(all_data$timestamp_first_active)
all_data$ts_first_active_month<-month(all_data$timestamp_first_active)
all_data$ts_first_active_day<-day(all_data$timestamp_first_active)
all_data$ts_first_active_wday<-wday(all_data$timestamp_first_active)
all_data$timestamp_first_active<-NULL


# One factor level is missing in first_affiliate_tracked
str(all_data)  # First factor level of first_affiliate_tracked is "", assign something to it
levels(all_data$first_affiliate_tracked)[1]<-"missing"

# cleaning Age data

#Setting all values above 100 and below 5 to NA's
all_data$age[all_data$age >= 100]<- NA
all_data$age[all_data$age <= 5]<- NA

#Idea is to replace NA's with random values between Mean-SD and Mean+SD
set.seed(10112017)

#Generating random values between Mean-SD and Mean+SD
random=trunc(runif(90418, min=mean(all_data$age,na.rm=TRUE)-sd(all_data$age,na.rm=TRUE)-1, max=mean(all_data$age,na.rm=TRUE)+sd(all_data$age,na.rm=TRUE)+1))

#Replacing NA's with random values between Mean-SD and Mean+SD
all_data$age[is.na(all_data$age)]<-random

# Splitting train and test data
X <-all_data %>% filter (IDA < 300000) %>% mutate(IDA = NULL) 

# Also merge in X the class variable: labels
X<-merge(X,labels,by='id')
trainindex<-createDataPartition(X$country_destination,p=0.8,list=FALSE)
training<-X[trainindex,]
test<-X[-trainindex,]

#Building a random forest model

m=randomForest(country_destination~age+gender+language+affiliate_channel+account_created_year+signup_flow+signup_app,data=training,ntree=750,mtry=7,importance=F,nfeatures=F,ncores=C)

#Validating the model with the training data
v_pred1 <- predict(m, training[,-1])
com1<-data.frame(predicted=v_pred1,actual=training$country_destination)
accuracy1<-sum(com1$predicted == com1$actual)/nrow(com1)

#accuracy1
#[1] 0.6256654

#summary(m)

# Performing Classification on testing data
X_test1 <- all_data %>% filter (IDA >= 300000) %>% mutate (IDA = NULL)
y_pred1 <- predict(m, X_test1[,-1])

# submit results
submit1<-data.frame(id=X_test1$id,country=y_pred)
write.csv(submit1, "submissionrf.csv", quote=FALSE, row.names = FALSE)

