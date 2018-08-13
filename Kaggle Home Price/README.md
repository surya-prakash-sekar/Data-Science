Kaggle - House Prices: Advanced Regression Techniques

Description

In this notebook we get a chance to explore and experiment with the Ames Housing 
dataset that was compiled by Dean De Cock for use in data science education. 
It's an incredible alternative for data scientists looking for a modernized and 
expanded version of the often cited Boston Housing dataset which is an 
introductory datset for ML Learners.

This Kaggle competition's dataset proves that much more influences the sales 
price negotiations than the number of bedrooms or a white-picket fence.
There are a total of 79 explanatory variables describing (almost) every aspect 
of residential homes in Ames, Iowa, this competition challenges us to predict 
the final price of each home.

Goal

It is our job to predict the sales price for each house. For each Id in the test 
set, we must predict the value of the SalePrice variable. 

Metric

Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the 
logarithm of the predicted value and the logarithm of the observed sales price.

Dependencies

This project requires Python 2.7 and the following Python libraries installed:

Scikit Learn
Seaborn
Numpy
Pandas
Matplotlib
Math


You will also need to have software installed to run and execute a Jupyter 
Notebook.

Result

Gradient Boosting Regressor had the best performance with an accuracy of 90% 
and a Logarithmic RMSE of over 0.0044.

