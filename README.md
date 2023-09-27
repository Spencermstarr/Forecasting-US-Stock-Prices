# Forecasting-US-Stock-Prices
This project is being done as an assigned individual project for a graduate course in 
big data analytics in my Masters of Science program in Data Analytics Engineering at 
George Mason University.

My goal here is to practice working with several of the fundamental and widely used classification models in machine/statistical learning that 
I have learned in this master's program thus far (as this is the penultimate course) by comparing their performance in terms of how well they are 
able to forecast the performance of US Stocks. In this context, because it is a classification task, that peformance is simply whether the price of
a given stock goes up or down in a given year.

I used a high usability rated Kaggle dataset with annual data on US stocks with over 200 candidate
predictors for each of an average of 4k stocks for 5 separate years from 2014 to 2018. 
Here is the link to that Kaggle dataset: https://www.kaggle.com/datasets/cnic92/200-financial-indicators-of-us-stocks-20142018
But mainly, I just stick with using the stock data for the years 2014 and 2015.

In the first version (in an R script titled 'Team4_Final_Code (with a RF & a Meta (Learning) Model added on)'), I compare 7 singular ML classification algorithms: 
Logit, PLS-DA, Elastic Net, a single Artificial Neural Network, MARS (Multivariate Adaptive Regression Splines), SVM, KNN, and 3 ensemble learning models: 
an Average (of several) Neural Networks, Random Forest, and a custom, manually constructed Meta Learning Model.

In the second version of the script, called 'AIT 622 Big Data Analytics Project script', I instead  compare the performance of 6.5 classification predictive modeling methods, namely: 
Logit, PLS-DA, Elastic Net, a single Artificial Neural Network, an Average (of several) Neural Networks, Support Vector Machine, and K-Nearest Neighbors. 

The main difference between the two different R scripts is actually not which classification algorithms are implemented and compared, it is the way the missing values in the dataset 
are handled. In the first script, I simple remove all observations/rows with at least one missing value, i.e. one blank column. But because there are a very significant proportion 
of rows with at least one missing value (the 2014 dataset has 3808 rows and 225 columns and the 2015 dataset has 4120 rows and 225 columns, but after dropping all rows with a at least
one MA in them, the 2014 dataset only has 513 rows and the 2015 dataset only has 597), I created a second script which uses mean value imputation rather than dropping all rows with nulls.
This of course maintains the same number of rows in each dataframe as they start out with at the end of the preprocessing stage.




