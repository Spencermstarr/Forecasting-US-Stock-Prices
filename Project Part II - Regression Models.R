#### Spencer Marlen-Starr's AIT 622 Class Project code/script
#### Regression Models only
#### Regression Methods used: 
#### 1 - Partial Least Squares Regression
#### 2 - Ridge Regression
#### 3 - MARS
#### 4 - Artificial Neural Net
#### 4.5- AvgNNs
#### 5 - SVM
#### 5.5- Bagged SVMs
#### 6 - Random Forest Regression
#### 7 - Traditionally Selected Multiple Regression Model



##### Part 1: setting up the environment, and loading both the necessary 
#####         packages and all of the datasets
library(ggplot2)
library(lattice)

library(plyr)
library(dplyr)

library(vip)
library(caret)
library(lars)
library(nnet)

library(parallel)
library(foreach)
library(iterators)
library(doParallel)

library(forecast)
library(pls)
library(zoo)


#allow multithreaded operating
threads <- detectCores()
cluster <- makePSOCKcluster(threads)
registerDoParallel(cluster)


# find out which working directory R has defaulted to
getwd()
# use Ctrl+Shift+H to select the proper WD if necessary, for this script,
# the proper working directory will be the unzipped version of the zip folder
# we turned everything required of our final deliverable for this project in with
setwd("~/George Mason University folders (local)/AIT 622 - Determining Needs for Complex Big Data Systems/AIT 622 Individual Project")
getwd()


## load all of the datasets (for both regression & classification) into R 
data2014 <- read.csv("2014_Financial_Data.csv", header = TRUE)
data2015 <- read.csv("2015_Financial_Data.csv", header = TRUE)
data2016 <- read.csv("2016_Financial_Data.csv", header = TRUE)
data2017 <- read.csv("2017_Financial_Data.csv", header = TRUE)
data2018 <- read.csv("2018_Financial_Data.csv", header = TRUE)




##### Part 2: data cleaning, wrangling, and pre-processing
# remove all predictors with either zero or near zero variance
ZeroVar <- nearZeroVar(data2014)

data2014 <- data2014[ , -ZeroVar]
data2015 <- data2015[ , -ZeroVar]
data2016 <- data2016[ , -ZeroVar]
data2017 <- data2017[ , -ZeroVar]
data2018 <- data2018[ , -ZeroVar]




### assign all categorical variables to separate objects so
### that I can remove all non-numeric columns manually
# 2014
x2014 <- data2014$X
class2014 <- data2014$Class
sector2014 <- data2014$Sector
# 2015
x2015 <- data2015$X
class2015 <- data2015$Class
sector2015 <- data2015$Sector
# 2016
x2016 <- data2016$X
class2016 <- data2016$Class
sector2016 <- data2016$Sector
# 2017
x2017 <- data2017$X
class2017 <- data2017$Class
sector2017 <- data2017$Sector
# 2018
x2018 <- data2018$X
class2018 <- data2018$Class
sector2018 <- data2018$Sector


class2014 <- ifelse(class2014 == 1, "Increase", "Decrease")
class2015 <- ifelse(class2015 == 1, "Increase", "Decrease")
class2016 <- ifelse(class2016 == 1, "Increase", "Decrease")
class2017 <- ifelse(class2017 == 1, "Increase", "Decrease")
class2018 <- ifelse(class2018 == 1, "Increase", "Decrease")

# convert the integer values in the Class column of both yearly stock market 
# datasets (which are both stored in their own dataframe) into factors stored in
# their own newly created objects separate from the dataframes they came from
class2014 <- as.factor(class2014)
class2015 <- as.factor(class2015)
class2016 <- as.factor(class2016)
class2017 <- as.factor(class2017)
class2018 <- as.factor(class2018)

# remove the X, Class, & Sector columns from each annual dataframe 'manually'
data2014 <- subset(data2014, select = -c(X, Class, Sector))
data2015 <- subset(data2015, select = -c(X, Class, Sector))
data2016 <- subset(data2016, select = -c(X, Class, Sector))
data2017 <- subset(data2017, select = -c(X, Class, Sector))
data2018 <- subset(data2018, select = -c(X, Class, Sector))



### interpolate the missing values in data2014
# find and count the position of all NAs (column-wise)
which(is.na(data2014))
# find all rows with at least 1 NA
which(rowSums(is.na(data2014)) != 0)
# count all NAs in data2014
sum(is.na(data2014))
# find/count the # of NAs in each column of data2014
colSums(is.na(data2014))

mean_data2014 <- lapply(na.omit(data2014), mean)
# mean_data2014 <- lapply(mean_data2014, round)
for(i in 1:length(data2014)) {
  data2014[is.na(data2014[, i]), i] <- mean(data2014[, i], na.rm = TRUE) }
# count all NAs in data2014
sum(is.na(data2014))
# now remove the mean_data2014 list since its purpose has been served
rm(mean_data2014)

# interpolate the missing values in data2015
mean_data2015 <- lapply(na.omit(data2015), mean)
for(i in 1:length(data2015)) {
  data2015[is.na(data2015[, i]), i] <- mean(data2015[, i], na.rm = TRUE) }
rm(mean_data2015)

# interpolate the missing values in data2016
mean_data2016 <- lapply(na.omit(data2016), mean)
for(i in 1:length(data2016)) {
  data2016[is.na(data2016[, i]), i] <- mean(data2016[, i], na.rm = TRUE) }
rm(mean_data2016)

# interpolate the missing values in data2017
mean_data2017 <- lapply(na.omit(data2017), mean)
for(i in 1:length(data2017)) {
  data2017[is.na(data2017[, i]), i] <- mean(data2017[, i], na.rm = TRUE) }
rm(mean_data2017)

# interpolate the missing values in data2018
sum(is.na(data2018))
mean_data2018 <- lapply(na.omit(data2018), mean)
for(i in 1:length(data2018)) {
  data2018[is.na(data2018[, i]), i] <- mean(data2018[, i], na.rm = TRUE) }
# count all NAs in data2018
sum(is.na(data2018))
# remove the mean_data2014 list since its purpose has already been served
rm(mean_data2018)


### Dealing with collinearity and multicollinearity:
## remove stock price variance column to prevent perfect multicollinearity...
pr_var2015 <- data2014$X2015.PRICE.VAR....
pr_var2016 <- data2015$X2016.PRICE.VAR....
pr_var2017 <- data2016$X2017.PRICE.VAR....
pr_var2018 <- data2017$X2018.PRICE.VAR....
pr_var2019 <- data2018$X2019.PRICE.VAR....
data2014 <- subset(data2014, select = -c(X2015.PRICE.VAR....))
data2015 <- subset(data2015, select = -c(X2016.PRICE.VAR....))
data2016 <- subset(data2016, select = -c(X2017.PRICE.VAR....))
data2017 <- subset(data2017, select = -c(X2018.PRICE.VAR....))
data2018 <- subset(data2018, select = -c(X2019.PRICE.VAR....))


## find and remove all predictors which are highly correlated in 
## the 2014 dataset from every dataset
correlations_R <- cor(data2014)

highCorr_R <- findCorrelation(correlations_R, cutoff = .8)

data2014 <- data2014[, -highCorr_R]
data2015 <- data2015[, -highCorr_R]
data2016 <- data2016[, -highCorr_R]
data2017 <- data2017[, -highCorr_R]
data2018 <- data2018[, -highCorr_R]



## reformat every dataset as a matrix for later use
data2014matrix <- as.matrix(data2014)
data2015matrix <- as.matrix(data2015)
data2016matrix <- as.matrix(data2016)
data2017matrix <- as.matrix(data2017)
data2018matrix <- as.matrix(data2018)


## Just in case, ya know?!
data2014r <- cbind(data2014, pr_var2014)
data2015r <- cbind(data2015, pr_var2015)
data2016r <- cbind(data2016, pr_var2016)
data2017r <- cbind(data2017, pr_var2017)
data2018r <- cbind(data2018, pr_var2018)s




# define model controls








###### Part II: Regression Modeling of the US Stock Data
######          Regression Methods used: Partial Least Squares Regression, 
######          Ridge Regression, MARS, Artificial Neural Networks,
######          Average Neural Networks, SVM, Random Forest Regression, 
######          and a custom MLR Specification I constructed myself


### Regression Forecasting Model #1: Partial Least Squares Regression
library(pls)
pls1Fit <- 


  

### Regression Forecasting Model #2: Ridge Regression
## method 1 (of 2): using the glmnet package
library(Matrix)
library(glmnet)
#grid1 <- 
ridge1.model = cv.glmnet(x = data2014matrix, y = pr_var2014, type.measure = "mse",
                         alpha = 0, family = "gaussian", standardize = TRUE)
dim(coef(ridge1.model))
class(ridge1.model)

# measure the test set accuracy of the predictions made by a ridge regression 
# trained on the entire 2014 stock market dataset in terms of how well they
# predict the stock market behavior of the entire 2015 dataset
ridge1.pred = predict(ridge1.model, s = ridge1.model$lambda.min,
                      newx = as.matrix(data2015))
ridge1.pred
dim(ridge1.pred)
length(pr_var2014)
mean((pr_var2014 - sample(ridge1.pred, 3808))^2)


## method 2 (of 2): using the MASS & elasticnet packages
library(MASS)
library(elasticnet)
ridge1 <- enet(x = as.matrix(data2014), y = pr_var2014, lambda = 1)
ridge1
summary(ridge1)
class(ridge1)

ridge1Preds <- predict(ridge1, newx = as.matrix(data2015), s = 1, 
                       mode = "fraction")
ridge1Preds
length(ridge1Preds$fit)
mean((pr_var2014 - sample(ridge1Preds$fit, 3808))^2)






### Regression Forecasting Model #3: Multivariate Adaptive Regression Splines
library(earth)
library(Formula)
library(plotmo)
library(plotrix)
library(earth)
marsGrid = expand.grid(.degree = 1:2, .nprune = 2:38)
set.seed(100)
# try using the train function from the caret package, and  
# setting the method argument equal to "earth".
marsModelR1 = train(x = data2014, y = pr_var2014, method = "earth", 
                    preProc = c("center", "scale"), tuneGrid = marsGrid)
marsModelR1

# compare the expected classifications in 2015 to the observed classifications in 2015
marsR1Pred2015 = predict(marsModelR1, newdata = data2015)
length(marsR1Pred)
dim(marsR1Pred)
str(marsR1Pred)
marsR1Pred_2016 = predict(marsModelR1, newdata = data2016)  # same as above for 2016
marsR1Pred_2017 = predict(marsModelR1, newdata = data2017)  # same as above for 2017
marsR1Pred_2018 = predict(marsModelR1, newdata = data2018)  # same as above for 2018

marsR1_PR = postResample(pred = marsR1Pred, obs = pr_var2014)
marsR1_PR


# try using the earth function from the earth package instead
marsFits_2014 = earth(x = data2014, y = pr_var2014, penalty = 3)
marsFits_2014
summary(marsFits_2014)

marsPred_2014 = predict(marsFits_2014, newdata = data2015)
marsPred_2014

marsPR_2014 = postResample(pred = marsFits_2014, obs = pr_var2014)
marsPR_2014






### Regression Forecasting Model #4: Artificial Neural Network
library(nnet)

set.seed(100)
# exact same grid (for now)
NNetR1_Grid = expand.grid(.decay = c(0, 0.01, 0.1), .size = 1:10)

# fit/run a NN with only 1 hidden layers
set.seed(100)
NNet_R1_1layer = train(data = data2014, formula = pr_var2014 ~ .,
                       x = data2014, y = pr_var2014, method = "nnet", 
                       preProc = c("center", "scale"), linout = TRUE, 
                       trace = FALSE, size = 2, rang = 0.15,
                       MaxNWts = 10 * (ncol(data2014) + 1) + 10 + 1, 
                       maxit = 200, tuneGrid = NNetR1_Grid)
NNet_R1_1layer
# fit/run a NN with only 2 hidden layers
set.seed(100)
NNet_R1_2layers = train(data = data2014, formula = pr_var2014 ~ .,
                        x = data2014, y = pr_var2014, method = "nnet", size = 2,
                    preProc = c("center", "scale"), linout = TRUE, trace = FALSE, 
                    MaxNWts = 10 * (ncol(data2014) + 1) + 10 + 1, maxit = 500, 
                    tuneGrid = nnR1_Grid)
NNet_R1_2layers



# fit/run a NN with 3 hidden layers
set.seed(100)
NNet_R1_3layers = train(x = data2014, y = pr_var2014, method = "nnet", size = 3,
                    preProc = c("center", "scale"), linout = FALSE, 
                    trace = FALSE, MaxNWts = 10 * (ncol(data2014) + 1) + 10 + 1, 
                    maxit = 500, tuneGrid = nnR1_Grid)


# fit/run a NN with only 4 hidden layers
set.seed(100)
NNet_R1_4layers = train(x = data2014, y = pr_var2014, method = "nnet", size = 4,
                    preProc = c("center", "scale"), linout = FALSE, 
                    trace = FALSE, MaxNWts = 10 * (ncol(data2014) + 1) + 10 + 1, 
                    maxit = 500, tuneGrid = nnR1_Grid)




### Regression Forecasting Model #4.5: Average (of several) Neural Networks
## Using an Average of several ANNs model trained on the 2014 stock market data
## to predict the performance of those same stocks in 2015.
# 1st method of fitting/estimation, using the train() function WITHOUT BAGGING
set.seed(100)
avgNNetModel_R1 = train(x = data2014, y = pr_var2014, method = "avNNet", bag = FALSE,
                      preProc = c("center", "scale"), linout = FALSE, trace = FALSE, 
                      MaxNWts = 10 * (ncol(data2014) + 1) + 10 + 1, maxit = 500)
avgNNetModel_R1
summary(avgNNetModel_R1)
coef(avgNNetModel_R1)


# 2nd method of fitting/estimation, using the avNNet() function 
# WITHOUT BAGGING, but with repeats

set.seed(100)
avgNNetModelR1_Method2 <- avNNet(x = data2014, y = pr_var2014, bag = FALSE,
                                  data = data2014r, size = 5, repeats = 5)
avgNNetModelR1_Method2
summary(avgNNetModelR1_Method2)
avgNNetModelR1_Method2$model
predict(avgNNetModelR1_Method2, data2015)

avgNNetModelR1_Method2_V2 <- avNNet(x = data2014, y = pr_var2014, data = data2014r, 
                             size = 5, bag = FALSE, repeats = 5, newdata = data2015)


# Using an Average of several ANNs model trained on the 2014 stock market data
# to predict the performance of those same stocks in 2016.
set.seed(100)
avgNNetModel_R2 = train(x = data2014, y = pr_var2014, method = "avNNet", bag = FALSE,
                        preProc = c("center", "scale"), linout = FALSE, trace = FALSE, 
                        MaxNWts = 10 * (ncol(data2014) + 1) + 10 + 1, maxit = 500)
avgNNetModel_R2




### Regression Forecasting Model #5: Support Vector Machine Regression
set.seed(100)




### Regression Forecasting Model #5.5: Bagged SVMs
set.seed(100)




## "Recall that bagging is simply a special case of a random forest with m = p.
##  Therefore, the randomForest() function can be used to perform both 
##  random forests and bagging. 
### Regression Forecasting Model #6: Random Forest Regression
library(randomForest)

set.seed(11)
RFR_R1 = randomForest(pr_var2014 ~ ., data = data2014, ntree = 110) 
class(RFR_R1)

yhat.RFR_R1 = predict(RFR_R1, newdata = data2015)
mean((yhat.RFR_R1 - as.matrix(data2015))^2)
# quantify & assess variable importance!
importance(RFR_R1)
# create a Variable Importance Plot
varImpPlot(RFR_R1)


RFreg.2014 = randomForest(pr_var2014 ~ ., data = data2014, ntree = 777, 
                          importance = TRUE)
class(RFreg.2014)

yhat.RFreg.2014 = predict(RFreg.2014, newdata = data2015)
mean((yhat.RFreg.2014 - data2015matrix)^2)  
# quantify & assess variable importance!
importance(RFreg.2014)
# create a Variable Importance Plot
varImpPlot(RFreg.2014)





### Regression Forecasting Model #7: Traditional Multiple Regression Model