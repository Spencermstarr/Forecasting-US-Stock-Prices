#### Spencer Marlen-Starr's AIT 622 Big Data Analytics Project

##### Part 1: setting up the environment, and loading both the necessary 
#####         packages and all of the datasets
# load all necessary packages using only 1 command
library_list <- c(library(ggplot2),library(lattice),library(plyr),
                  library(dplyr),library(readr),library(vip),library(caret),
                  library(pROC),library(nnet),library(parallel),
                  library(foreach),library(iterators),library(doParallel),
                  library(forecast),library(pls),library(zoo))


library(ggplot2)
library(lattice)

library(plyr)
library(dplyr)
library(readr)

library(vip)
library(caret)
library(pROC)
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
#setwd("~/George Mason University folders (local)/AIT 622 - Determining Needs for Complex Big Data Systems/AIT 622 Individual Project")
#setwd("C:/Users/Spencer/OneDrive/Documents/Analytics Projects/AIT 622 Individual Project")
#getwd()



## load all of the datasets (for both regression & classification) into R 
## using the read_csv() function from the readr library from R's tidyverse
data_2014 <- read_csv("2014_Financial_Data.csv", col_names = TRUE)
data_2015 <- read_csv("2015_Financial_Data.csv", col_names = TRUE)
data_2016 <- read_csv("2016_Financial_Data.csv", col_names = TRUE)
data_2017 <- read_csv("2017_Financial_Data.csv", col_names = TRUE)
data_2018 <- read_csv("2018_Financial_Data.csv", col_names = TRUE)



##### Part 2: data cleaning, wrangling, and pre-processing
# remove all predictors with either zero or near zero variance 
# in the data_2014 dataset
ZeroVar2014 <- nearZeroVar(data_2014)

data_2014 <- data_2014[ , -ZeroVar2014]
data_2015 <- data_2015[ , -ZeroVar2014]
data_2016 <- data_2016[ , -ZeroVar2014]
data_2017 <- data_2017[ , -ZeroVar2014]
data_2018 <- data_2018[ , -ZeroVar2014]



# remove all rows with missing values from the 2014 financial dataset
mean(is.na(data_2014))
data_2014c <- na.omit(data_2014)
count(data_2014c, "Class")
dim(data_2014c)
length(data_2014c)
mean(is.na(data_2014c))

data_2015c <- na.omit(data_2015)  # same same for the 2015 financial dataset
count(data_2015c, "Class")
dim(data_2015c)

data_2016c <- na.omit(data_2016)  # same same for the 2016 financial dataset

data_2017c <- na.omit(data_2017)  # same same for the 2017 financial dataset

# remove all rows with missing values from the 2018 financial dataset
mean(is.na(data_2018))
data_2018c <- na.omit(data_2018)
count(data_2018c, "Class")
dim(data_2018c)
length(data_2018c)
mean(is.na(data_2018c))



### assign all categorical variables to separate objects so
### that I can remove all non-numeric columns manually
# 2014
x_2014 <- data_2014$...1
class_2014 <- data_2014$Class
sector_2014 <- data_2014$Sector
x_2014c <- data_2014c$X
class_2014c <- data_2014c$Class
sector_2014c <- data_2014c$Sector
# 2015
x_2015 <- data_2015$...1
class_2015 <- data_2015$Class
sector_2015 <- data_2015$Sector
x_2015c <- data_2015c$...1
class_2015c <- data_2015c$Class
sector_2015c <- data_2015c$Sector
# 2016
x_2016 <- data_2016$...1
class_2016 <- data_2016$Class
sector_2016 <- data_2016$Sector
x_2016c <- data_2016c$...1
class_2016c <- data_2016c$Class
sector_2016c <- data_2016c$Sector
# 2017
x_2017 <- data_2017$...1
class_2017 <- data_2017$Class
sector_2017 <- data_2017$Sector
x_2017c <- data_2017c$...1
class_2017c <- data_2017c$Class
sector_2017c <- data_2017c$Sector
# 2018
x_2018 <- data_2018$...1
class_2018 <- data_2018$Class
sector_2018 <- data_2018$Sector
x_2018c <- data_2018c$...1
class_2018c <- data_2018c$Class
sector_2018c <- data_2018c$Sector


# use ifelse functions to distinguish between whether the price of 
# a stock increased or decreased during a given year using those words
class_2014 <- ifelse(class_2014 == 1, "Increase", "Decrease")
class_2015 <- ifelse(class_2015 == 1, "Increase", "Decrease")
class_2016 <- ifelse(class_2016 == 1, "Increase", "Decrease")
class_2017 <- ifelse(class_2017 == 1, "Increase", "Decrease")
class_2018 <- ifelse(class_2018 == 1, "Increase", "Decrease")

class_2014c <- ifelse(class_2014c == 1, "Increase", "Decrease")
class_2015c <- ifelse(class_2015c == 1, "Increase", "Decrease")
class_2016c <- ifelse(class_2016c == 1, "Increase", "Decrease")
class_2017c <- ifelse(class_2017c == 1, "Increase", "Decrease")
class_2018c <- ifelse(class_2018c == 1, "Increase", "Decrease")


# convert the integer values in the class column of both yearly stock market 
# datasets (which are both stored in their own dataframe) into factors stored in
# their own newly created objects separate from the dataframes they came from
class_2014 <- as.factor(class_2014)
class_2015 <- as.factor(class_2015)
class_2016 <- as.factor(class_2016)
class_2017 <- as.factor(class_2017)
class_2018 <- as.factor(class_2018)

class_2014c <- as.factor(class_2014c)
class_2015c <- as.factor(class_2015c)
class_2016c <- as.factor(class_2016c)
class_2017c <- as.factor(class_2017c)
class_2018c <- as.factor(class_2018c)

# remove the X, Class, & Sector columns from each annual dataframe 'manually'
data_2014 <- subset(data_2014, select = -c(...1, Class, Sector))
data_2015 <- subset(data_2015, select = -c(...1, Class, Sector))
data_2016 <- subset(data_2016, select = -c(...1, Class, Sector))
data_2017 <- subset(data_2017, select = -c(...1, Class, Sector))
data_2018 <- subset(data_2018, select = -c(...1, Class, Sector))
data_2014c <- subset(data_2014c, select = -c(...1, Class, Sector))
data_2015c <- subset(data_2015c, select = -c(...1, Class, Sector))
data_2016c <- subset(data_2016c, select = -c(...1, Class, Sector))
data_2017c <- subset(data_2017c, select = -c(...1, Class, Sector))
data_2018c <- subset(data_2018c, select = -c(...1, Class, Sector))



### interpolate the missing values in data_2014 using their means
# find and count the position of all NAs (column-wise)
which(is.na(data_2014))
# find all rows with at least 1 NA
which(rowSums(is.na(data_2014)) != 0)
# count all NAs in data_2014
sum(is.na(data_2014))
# find/count the # of NAs in each column of data_2014
colSums(is.na(data_2014))

mean_data_2014 <- lapply(na.omit(data_2014), mean)
# mean_data_2014 <- lapply(mean_data_2014, round)
for(i in 1:length(data_2014)) {
  data_2014[is.na(data_2014[, i]), i] <- mean(data_2014[, i], na.rm = TRUE) }
# count all NAs in data_2014
sum(is.na(data_2014))
# now remove the mean_data_2014 list since its purpose has been served
rm(mean_data_2014)

# interpolate the missing values in data_2015
mean_data_2015 <- lapply(na.omit(data_2015), mean)
for(i in 1:length(data_2015)) {
  data_2015[is.na(data_2015[, i]), i] <- mean(data_2015[, i], na.rm = TRUE) }
rm(mean_data_2015)

# interpolate the missing values in data_2016
mean_data_2016 <- lapply(na.omit(data_2016), mean)
for(i in 1:length(data_2016)) {
  data_2016[is.na(data_2016[, i]), i] <- mean(data_2016[, i], na.rm = TRUE) }
rm(mean_data_2016)

# interpolate the missing values in data_2017
mean_data_2017 <- lapply(na.omit(data_2017), mean)
for(i in 1:length(data_2017)) {
  data_2017[is.na(data_2017[, i]), i] <- mean(data_2017[, i], na.rm = TRUE) }
rm(mean_data_2017)

# interpolate the missing values in data_2018
sum(is.na(data_2018))
mean_data_2018 <- lapply(na.omit(data_2018), mean)
for(i in 1:length(data_2018)) {
  data_2018[is.na(data_2018[, i]), i] <- mean(data_2018[, i], na.rm = TRUE) }
# count all NAs in data_2018
sum(is.na(data_2018))
# remove the mean_data_2014 list since its purpose has already been served
rm(mean_data_2018)



### Dealing with collinearity and multicollinearity:
## remove stock price variance column to prevent perfect multicollinearity...
# ... for all of the datasets I will use for my regressions
pr_var2015 <- data_2014$`2015 PRICE VAR [%]`
pr_var2016 <- data_2015$`2016 PRICE VAR [%]`
pr_var2017 <- data_2016$`2017 PRICE VAR [%]`
pr_var2018 <- data_2017$`2018 PRICE VAR [%]`
pr_var2019 <- data_2018$`2019 PRICE VAR [%]`
data_2014 <- subset(data_2014, select = -c(`2015 PRICE VAR [%]`))
data_2015 <- subset(data_2015, select = -c(`2016 PRICE VAR [%]`))
data_2016 <- subset(data_2016, select = -c(`2017 PRICE VAR [%]`))
data_2017 <- subset(data_2017, select = -c(`2018 PRICE VAR [%]`))
data_2018 <- subset(data_2018, select = -c(`2019 PRICE VAR [%]`))

# ... for all the datasets I will use for classification
pr_var2015c <- data_2014c$`2015 PRICE VAR [%]`
pr_var2016c <- data_2015c$`2016 PRICE VAR [%]`
pr_var2017c <- data_2016c$`2017 PRICE VAR [%]`
pr_var2018c <- data_2017c$`2018 PRICE VAR [%]`
pr_var2019c <- data_2018c$`2019 PRICE VAR [%]`
data_2014c <- subset(data_2014c, select = -c(`2015 PRICE VAR [%]`))
data_2015c <- subset(data_2015c, select = -c(`2016 PRICE VAR [%]`))
data_2016c <- subset(data_2016c, select = -c(`2017 PRICE VAR [%]`))
data_2017c <- subset(data_2017c, select = -c(`2018 PRICE VAR [%]`))
data_2018c <- subset(data_2018c, select = -c(`2019 PRICE VAR [%]`))


## find and remove all predictors which are highly correlated in 
## the 2014 dataset from every dataset
#correlations_R <- cor(data_2014)
#correlations_R_no_missing <- cor(na.omit(correlations_R))
#correlations_C <- cor(data_2014c)

# I arbitrarily define highly correlated as above 80% here
#highCorr_R <- findCorrelation(correlations_R_no_missing, cutoff = 0.8)
#highCorr_C <- findCorrelation(correlations_C, cutoff = 0.8)

## find and remove all predictors which are highly correlated in 
## the datasets used for regression
# Calculate correlation matrix for each year using pairwise complete observations
correlations_2014 <- cor(data_2014, use = "pairwise.complete.obs")
correlations_2015 <- cor(data_2015, use = "pairwise.complete.obs")
correlations_2016 <- cor(data_2016, use = "pairwise.complete.obs")
correlations_2017 <- cor(data_2017, use = "pairwise.complete.obs")
correlations_2018 <- cor(data_2018, use = "pairwise.complete.obs")

highCorr_2014 <- findCorrelation(correlations_2014, cutoff = 0.8)
highCorr_2015 <- findCorrelation(correlations_2015, cutoff = 0.8)
highCorr_2016 <- findCorrelation(correlations_2016, cutoff = 0.8)
highCorr_2017 <- findCorrelation(correlations_2017, cutoff = 0.8)
highCorr_2018 <- findCorrelation(correlations_2018, cutoff = 0.8)

# now remove all highly correlated variables from the versions
# of the datasets to be used for regression modeling
data_2014 <- data_2014[, -highCorr_2014]
data_2015 <- data_2015[, -highCorr_2015]
data_2016 <- data_2016[, -highCorr_2016]
data_2017 <- data_2017[, -highCorr_2017]
data_2018 <- data_2018[, -highCorr_2018]


## find and remove all predictors which are highly correlated in 
## the datasets used for classification
# Calculate correlation matrix for each year using pairwise complete observations
correlations_2014C <- cor(data_2014c, use = "pairwise.complete.obs")
correlations_2015C <- cor(data_2015c, use = "pairwise.complete.obs")
correlations_2016C <- cor(data_2016c, use = "pairwise.complete.obs")
correlations_2017C <- cor(data_2017c, use = "pairwise.complete.obs")
correlations_2018C <- cor(data_2018c, use = "pairwise.complete.obs")

highCorr_2014C <- findCorrelation(correlations_2014C, cutoff = 0.8)
highCorr_2015C <- findCorrelation(correlations_2015C, cutoff = 0.8)
highCorr_2016C <- findCorrelation(correlations_2016C, cutoff = 0.8)
highCorr_2017C <- findCorrelation(correlations_2017C, cutoff = 0.8)
highCorr_2018C <- findCorrelation(correlations_2018C, cutoff = 0.8)


# now remove all highly correlated variables from the versions
# of the datasets to be used for regression modeling
data_2014c <- data_2014c[, -highCorr_2014C]
data_2015c <- data_2015c[, -highCorr_2015C]
data_2016c <- data_2016c[, -highCorr_2016C]
data_2017c <- data_2017c[, -highCorr_2017C]
data_2018c <- data_2018c[, -highCorr_2018C]




# define model controls
ctrl_c <- trainControl(method = "LGOCV", summaryFunction = twoClassSummary,
                     classProbs = TRUE)









##### Part 3: Forecasting US Stock Behavior via Classification Modeling:
#####         Creating 6.5 classification forecasting models which generate 
#####         objectively comparable predictions. The 6.5 classification 
#####         predictive modeling methods I chose are: Logit, PLS-DA,
#####         Elastic Net, an Artificial Neural Network, an Average (of several) 
#####         Neural Networks, Support Vector Machines, and KNN.
#####         Important note: when our comments say that some predict() is 
#####         comparing the expected classification in 2015 (given a model 
#####         trained on the data from 2014) vs the observed classification 
#####         in 2015, that just means the degree to which stocks that 
#####         our model predicted would go up (in 2015 based on how 
#####         it was trained & cross-validated on the 2014 data) actually
#####         went up. And likewise, the degree to which stocks our model 
#####         predicted would go down in 2015 actually did go down.


### Classification Forecasting Model #1: Logit
# set a random seed to ensure the replicability of our predictions
set.seed(100)          # use the same seed for every model
ftLogitC1 <- train(x = data_2014c, y = class_2014c, method = "glm",
                 metric = "ROC", preProcess = c("center", "scale"),
                 trControl = ctrl_c)
ftLogitC1

# compare the expected classifications in 2015 to the observed classifications in 2015
LogitC1_predictions_for_2015 <- predict(ftLogitC1, data_2015c)
LogitC1_predictions_for_2016 <- predict(ftLogitC1, data_2016c)   # same same for 2016
LogitC1_predictions_for_2016 <- predict(ftLogitC1, data_2017c)   # same same for 2017

# create a confusion matrix to show how well our Logit model's predictions fared
# by inspecting all of its classification model performance metrics
LogitC1_CFM <- confusionMatrix(data = LogitC1_predictions_for_2015, 
                               reference = class_2015c, positive = "Increase")
LogitC1_CFM

LogitC1_prob2015 <- predict(ftLogitC1, data_2015c, type = "prob")
LogitC1_prob2016 <- predict(ftLogitC1, data_2016c, type = "prob")
LogitC1_prob2017 <- predict(ftLogitC1, data_2018c, type = "prob")


# create an ROC curve for our Logit model and store it in an object
ROC_LogitC1 <- roc(response = class_2015c, predictor = LogitC1_prob2015$Increase,
                 levels = rev(levels(class_2015c)))
# plot that ROC curve
plot(ROC_LogitC1, col = "red", lwd = 3, 
     main = "ROC curve for the Logit fit on the reduced 2014 stock data,
     the version with all missing valued rows omitted")

# calculate the area under the ROC curve
auc_C1 <- auc(ROC_LogitC1)
cat('Area under the ROC curve for Logistic Regression: ', round(auc_C1, 3))

# assess the importance of the included regressors  
VarImp_LogitC1 <- varImp(ftLogitC1)
VarImp_LogitC1

# create a variable importance plot (for only the top 10 most important regressors)
plot(VarImp_LogitC1, top = 10, 
     main = 'Variable Importance Plot for my Logit Model fit on the 2014 stock data')




### Classification Forecasting Model #2: Partial Least Squares Discriminant Analysis
set.seed(100)   # use the same seed for every model
ftPLS_C1 <- train(x = data_2014c, y = class_2014c, method = "pls",
               tuneGrid = expand.grid(.ncomp = 1:4),
               preProc = c("center", "scale"), metric = "ROC",
               trControl = ctrl_c)
ftPLS_C1

# compare the expected classifications in 2015 to the observed classifications in 2015
PLS_C1predict <- predict(ftPLS_C1, data_2015c)

# create a confusion matrix to show how well our PLS-DA model's predictions fared
# by inspecting all of its classification model performance metrics
PLSDA_C1_CFM <- confusionMatrix(data = PLS_C1predict, reference = class_2015c,
                                positive = "Increase")
PLSDA_C1_CFM

PLSDA_C1prob2015 <- predict(ftPLS_C1, data_2015c, type = "prob")
PLSDA_C1prob2017 <- predict(ftPLS_C1, data_2017c, type = "prob")

# create an ROC curve for our PLS-DA model and store it in an object
rocPLSDA_C1 <- roc(response = class_2015c, predictor = PLSDA_C1prob2015$Increase,
              levels = rev(levels(class_2015c)))

# show the plot of that ROC curve for our PLS regression
plot(rocPLSDA_C1, col = "red", lwd = 3, 
     main = "ROC curve for the PLS-DA fit on the reduced 2014 stock data, 
     the version of it with all rows with NAs omitted")

auc_C2 <- auc(rocPLSDA_C1)
cat('Area under the ROC curve for the PLS-DA model fit on the redueced: ', 
    round(auc_C2, 3))

# assess the importance of the included regressors  
VarImp_PLSDA_C1 <- varImp(ftPLS_C1)
VarImp_PLSDA_C1

# create a variable importance plot (for only the top 20 IVs)
plot(VarImp_PLSDA_C1, top = 10, 
     main = 'Variable Importance Plot for our PLS-DA model fit on the 2014 data')





### Classification Forecasting Model #3: penalized regression models
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1), 
                        .lambda = seq(.01, .2, length = 40))

set.seed(100)        # use the same seed for every predictive model
glmnTunedC1 <- train(x = data_2014c, y = class_2014c, method = "glmnet",
                   tuneGrid = glmnGrid, preProc = c ("center", "scale"),
                   metric = "ROC", trControl = ctrl_c)
glmnTunedC1

glmnPredictC1 <- predict(glmnTunedC1, data_2015c)
#glmnPredictC1_2016 <- predict(glmnTunedC1, data_2016c)
#glmnPredictC1_2017 <- predict(glmnTunedC1, data_2017c)

# assess the performance of our penalized classification model's 
# predictions via confusion matrix
PRM_CFM_C1 <- confusionMatrix(data = glmnPredictC1, 
                           reference = class_2015c,
                           positive = "Increase")
PRM_CFM_C1

glmnProb <- predict(glmnTunedC1, data_2015c, type = "prob")

# create an ROC curve for our penalized classification model
glmnROC_C1 <- roc(response = class_2015c, predictor = glmnProb$Increase,
               levels = rev(levels(class_2015c)))

# plot the ROC curve
plot(glmnROC_C1, col = "red", lwd = 3, 
     main = "ROC curve for elastic net fit on the reduced 2014 stock data")

auc_C3 <- auc(glmnROC_C1)
cat('Area under the ROC curve for our penalized model: ', 
    round(auc_C3, 4))

# assess the importance of the included regressors  
impGLMN <- varImp(glmnTunedC1)
impGLMN

# create a variable importance plot (for only the top 20 regressors)
plot(impGLMN, top = 10, main = 'Variable Importance Plot for our penalized model')





### Classification Forecasting Model #4: Artificial Neural Network
nnGrid = expand.grid(.decay = c(0, 0.01, 0.1), .size = 1:10)

set.seed(100)
nnetModelC1 = train(x = data_2014c, y = class_2014c, method = "nnet", 
                  preProc = c("center", "scale"), linout = FALSE, trace = FALSE, 
                  MaxNWts = 10 * (ncol(data_2014c) + 1) + 10 + 1, 
                  maxit = 500, tuneGrid = nnGrid)
nnetModelC1

# compare the expected classifications in 2015 to the 
# observed classifications in 2015
nnetPredC1 = predict(nnetModelC1, newdata = data_2015c)

nnetPRc1 = postResample(pred = nnetPredC1, obs = class_2015c)
nnetPRc1

# assess the performance of our Neural Network's predictions for the 2015 test set
# via a confusion matrix
nnet_CFM_C1 <- confusionMatrix(data = nnetPredC1, reference = class_2015c,
                               positive = "Increase")
nnet_CFM_C1

nnetC1_Prob <- predict(nnetModelC1, data_2015c, type = "prob")

nnetC1_ROC <- roc(response = class_2015c, predictor = nnetC1_Prob$Increase,
               levels = rev(levels(class_2015c)))

plot(nnetC1_ROC, col = "red", lwd = 3, 
     main = "ROC curve for the Neural Network fit on the reduced 2014 data")

# calculate the area under the above ROC curve
auc_C4 <- auc(nnetC1_ROC)
cat('Area under the ROC curve for our Neural Network:', round(auc_C4, 4))

# assess the importance of the included regressors  
impNNETc1 <- varImp(nnetModelC1)
impNNETc1

# create a variable importance plot
plot(impNNETc1, top = 10, main = 'Variable Importance for Neural Network')



## Classification Forecasting Model #4.5: Average Neural Network
set.seed(100)
avNNetModelC1 = train(x = data_2014c, y = class_2014c, method = "avNNet", 
                    preProc = c("center", "scale"), linout = FALSE, trace = FALSE, 
                    MaxNWts = 10 * (ncol(data_2014c) + 1) + 10 + 1, maxit = 500)
avNNetModelC1

# compare the expected classifications in 2015 to the observed classifications in 2015
avNNetC1_Pred = predict(avNNetModelC1, newdata = data_2015c)

avNNetC1_PR = postResample(pred = avNNetC1_Pred, obs = class_2015c)
avNNetC1_PR

# assess the performance of the predictions made by our 
# average neural nets model for the 2015 testing dataset via confusion matrix
avgNNc1_CFM <- confusionMatrix(data = avNNetC1_Pred, reference = class_2015c,
                             positive = "Increase")
avgNNc1_CFM
  
avNNetC1_Prob <- predict(avNNetModelC1, data_2015c, type = "prob")

avNNetModelC1_ROC <- roc(response = class_2015c, 
                         predictor = avNNetC1_Prob$Increase,
                         levels = rev(levels(class_2015c)))

plot(avNNetModelC1_ROC, col = "red", lwd = 3, 
     main = "ROC curve for our Average Neural Net")

# calculate the area under the above ROC curve
auc_C5 <- auc(avNNetModelC1_ROC)
cat('Area under the ROC curve for our Average Neural Net:', round(auc_C5, 4))

# assess the importance of the included regressors  
impAVNNet_c1 <- varImp(avNNetModelC1)
impAVNNet_c1

# create a variable importance plot
plot(impAVNNet_c1, top = 10, main = 'Variable Importance for Average Neural Network')










### Classification Forecasting Model #5: Support Vector Machine
library(kernlab)
set.seed(100)
svmGrid = expand.grid(.sigma = c(0, 0.01, 0.1), .C = 1:10)

svmC1.fit <- train(x = data_2014c, y = class_2014c,  method = "svmRadial",
                preProc = c("center", "scale"), tuneGrid = svmGrid,
                trControl = trainControl(method = "repeatedcv", 
                                         repeats = 5, classProbs =  TRUE))
svmC1.fit

# compare the expected classifications in 2015 to the observed classifications in 2015
svmC1_Preds = predict(svmC1.fit, newdata = data_2015c)

svmC1_PR = postResample(pred = svmC1_Preds, obs = class_2015c)
svmC1_PR

# assess the performance of our penalized classification model's predictions 
# via a confusion matrix
svmC1_CFM <- confusionMatrix(data = svmC1_Pred, reference = class_2015c,
                           positive = "Increase")
svmC1_CFM

svmC1Prob <- predict(svmC1Fit, data_2015c, type = "prob")

svmC1_ROC <- roc(response = class_2015c, predictor = svmC1Prob$Increase,
              levels = rev(levels(class_2015c)))

plot(svmC1_ROC, col = "red", lwd = 3, main = "ROC curve for our SVM model")

# calculate the area under the above ROC curve
auc_C7 <- auc(svmC1_ROC)
cat('Area under the ROC curve for the Support Vector Machine model: ', 
    round(auc_C7, 4))






### Classification Forecasting Model #6: K-Nearest Neighbors 
set.seed(100)
knnC1 = train(x = data_2014c, y = class_2014c, method = "knn",
                 preProc = c("center","scale"), tuneLength = 20)
knnC1

# compare the expected classifications in 2015 to the observed classifications in 2015
knnC1_Pred = predict(knnC1_Model, newdata = data_2015c)

knnC1_PR = postResample(pred = knnC1_Pred, obs = class_2015c)
knnC1_PR

knnC1_CFM <- confusionMatrix(data = knnC1_Pred, reference = class_2015c,
                             positive = "Increase")
knnC1_CFM

knnC1_Prob <- predict(knnC1_Model, data_2015c, type = "prob")

knnC1_ROC <- roc(response = class_2015c, predictor = knnC1_Prob$Increase,
              levels = rev(levels(class_2015c)))

plot(knnROC, col = "red", lwd = 3, main = "ROC curve the our KNN Model")

# calculate the area under the above ROC curve
auc_C8 <- auc(knnC1_ROC)
cat('Area under the ROC curve for the KNN model fit on the 2014 data: ', 
    round(auc_C8, 4))





#### Part 4: Classification Model Performance Assessment & Comparison
####         print out the Confusion Matrices for all 8 of our 
####         classification models again, but this time, print them 
####         out right on top of each other, one after the other in
####         order to facilitate easy visual comparisons of each of their
####         classification performance metrics with all the other 7.
LogitC1_CFM            # the confusion matrix for our logistic regression model

PLSDA_C1_CFM            # the confusion matrix for our PLS-DA model

PRM_C1_CFM              # the confusion matrix for our penalized regression model

nnet_CFM_C1          # the confusion matrix for our Neural Network model

avgNN_CFM_c1         # the confusion matrix for our average neural net

svmC1_CFM            # the confusion matrix for our Support Vector Machine model

knnC1_CFM            # the confusion matrix for our K-Nearest Neighbors model








###### Part 5: Forecasting US Stock Behavior via Regression Modeling
######         Regression Methods used: Partial Least Squares Regression, 
######         Ridge Regression, MARS, Artificial Neural Networks,
######         Average Neural Networks, Random Forest Regression, 
######         and a custom MLR Specification I constructed myself

### Regression Forecasting Model #1: Partial Least Squares Regression




### Regression Forecasting Model #2: Ridge Regression




### Regression Forecasting Model #3: Multivariate Adaptive Regression Splines
library(earth)
library(Formula)
library(plotmo)
library(plotrix)
library(earth)
marsGrid = expand.grid(.degree = 1:2, .nprune = 2:38)
## Try using the train function from the caret package, and  
## setting the method argument equal to "earth".
set.seed(100)
marsModelR1 = train(x = data_2014, y = pr_var2014, method = "earth", 
                    preProc = c("center", "scale"), tuneGrid = marsGrid)
marsModelR1

# compare the expected classifications in 2015 to the observed classifications in 2015
marsR1Pred2015 = predict(marsModelR1, newdata = data_2015)
length(marsR1Pred)
dim(marsR1Pred)
str(marsR1Pred)
marsR1Pred_2016 = predict(marsModelR1, newdata = data_2016)  # same as above for 2016
marsR1Pred_2017 = predict(marsModelR1, newdata = data_2017)  # same as above for 2017
marsR1Pred_2018 = predict(marsModelR1, newdata = data_2018)  # same as above for 2018

marsR1_PR = postResample(pred = marsR1Pred, obs = pr_var2014)
marsR1_PR


## Try using the earth function from the earth package instead.
marsFits_2014 = earth(x = data_2014, y = pr_var2014, penalty = 3)
marsFits_2014
summary(marsFits_2014)

marsPred_2015 = predict(marsFits_2014, newdata = data_2015)
marsPred_2015
length(marsR1Pred)
dim(marsR1Pred)
str(marsR1Pred)

marsPred_2016 = predict(marsFits_2015, newdata = data_2016)
marsPred_2016
marsPred_2017 = predict(marsFits_2017, newdata = data_2018)
marsPred_2017

mars2014_PR = postResample(pred = marsFits_2014, obs = pr_var2014)
mars2014_PR



### Regression Forecasting Model #4: Artificial Neural Network
library(nnet)

set.seed(100)
# exact same grid (for now)
nnR1_Grid = expand.grid(.decay = c(0, 0.01, 0.1), .size = 1:10)

set.seed(100)
nnetModelR1 = train(x = data_2014, y = class_2014, method = "nnet", 
                    preProc = c("center", "scale"), linout = FALSE, 
                    trace = FALSE, MaxNWts = 10 * (ncol(data_2014) + 1) + 10 + 1, 
                    maxit = 500, tuneGrid = nnR1_Grid)
nnetModelR1




### Regression Forecasting Model #4.5: Average (of several) Neural Networks
set.seed(100)
avNNetModel_R11 = train(x = data_2014, y = class_2014, method = "avNNet", 
                        preProc = c("center", "scale"), linout = FALSE, trace = FALSE, 
                        MaxNWts = 10 * (ncol(data_2014) + 1) + 10 + 1, maxit = 500)
avNNetModel_R1







### Regression Forecasting Model #5: Support Vector Machine Regression
library(e1071)
data_2014r <- cbind(data_2014, pr_var2014)
set.seed(100)
# figure out what to set the gamma & cost arguments to later!
svm_R1_Fits = svm(pr_var2014 ~ ., data = data_2014, scale = FALSE, 
                  kernel = "linear")
svm_R1_Fits <- svm(data_2014, pr_var2014)
summary(svm_R1_Fits)



### Regression Forecasting Model #5.5: Average (of several) Neural Networks






### Regression Forecasting Model #6: Random Forest Regression
library(randomForest)



