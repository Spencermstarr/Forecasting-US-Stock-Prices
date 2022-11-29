#### Spencer Marlen-Starr's AIT 622 Big Data Analytics Project
#### Classification Models only

##### Part 1: setting up the environment, and loading both the necessary 
#####         packages and all of the datasets
library(ggplot2)
library(lattice)

library(plyr)
library(dplyr)

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
# use ctrl_C+Shift+H to select the proper WD if necessary, for this script,
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



# remove all rows with missing values from the 2014 financial dataset
mean(is.na(data2014))
data2014c <- na.omit(data2014)
count(data2014c, "Class")
dim(data2014c)
length(data2014c)
mean(is.na(data2014c))

data2015c <- na.omit(data2015)  # same same for the 2015 financial dataset
count(data2015c, "Class")
dim(data2015c)

data2016c <- na.omit(data2016)  # same same for the 2016 financial dataset

data2017c <- na.omit(data2017)  # same same for the 2017 financial dataset

# remove all rows with missing values from the 2018 financial dataset
mean(is.na(data2018))
data2018c <- na.omit(data2018)
count(data2018c, "Class")
dim(data2018c)
length(data2018c)
mean(is.na(data2018c))



### assign all categorical variables to separate objects so
### that I can remove all non-numeric columns manually
# 2014
x2014c <- data2014c$X
class2014c <- data2014c$Class
sector2014c <- data2014c$Sector
# 2015
x2015c <- data2015c$X
class2015c <- data2015c$Class
sector2015c <- data2015c$Sector
# 2016
x2016c <- data2016c$X
class2016c <- data2016c$Class
sector2016c <- data2016c$Sector
# 2017
x2017c <- data2017c$X
class2017c <- data2017c$Class
sector2017c <- data2017c$Sector
# 2018
x2018c <- data2018c$X
class2018c <- data2018c$Class
sector2018c <- data2018c$Sector


class2014c <- ifelse(class2014c == 1, "Increase", "Decrease")
class2015c <- ifelse(class2015c == 1, "Increase", "Decrease")
class2016c <- ifelse(class2016c == 1, "Increase", "Decrease")
class2017c <- ifelse(class2017c == 1, "Increase", "Decrease")
class2018c <- ifelse(class2018c == 1, "Increase", "Decrease")

# convert the integer values in the Class column of both yearly stock market 
# datasets (which are both stored in their own dataframe) into factors stored in
# their own newly created objects separate from the dataframes they came from
class2014c <- as.factor(class2014c)
class2015c <- as.factor(class2015c)
class2016c <- as.factor(class2016c)
class2017c <- as.factor(class2017c)
class2018c <- as.factor(class2018c)

# remove the X, Class, & Sector columns from each annual dataframe 'manually'
data2014c <- subset(data2014c, select = -c(X, Class, Sector))
data2015c <- subset(data2015c, select = -c(X, Class, Sector))
data2016c <- subset(data2016c, select = -c(X, Class, Sector))
data2017c <- subset(data2017c, select = -c(X, Class, Sector))
data2018c <- subset(data2018c, select = -c(X, Class, Sector))



### Dealing with collinearity and multicollinearity:
## remove stock price variance column to prevent perfect multicollinearity
## for all the datasets I will use for classification
data2014c <- subset(data2014c, select = -c(X2015.PRICE.VAR....))
data2015c <- subset(data2015c, select = -c(X2016.PRICE.VAR....))
data2016c <- subset(data2016c, select = -c(X2017.PRICE.VAR....))
data2017c <- subset(data2017c, select = -c(X2018.PRICE.VAR....))
data2018c <- subset(data2018c, select = -c(X2019.PRICE.VAR....))


## find and remove all predictors which are highly correlated in 
## the 2014 dataset from every dataset
correlations_C <- cor(data2014c)

highCorr_C <- findCorrelation(correlations_C, cutoff = .8)

data2014c <- data2014c[, -highCorr_C]
data2015c <- data2015c[, -highCorr_C]
data2016c <- data2016c[, -highCorr_C]
data2017c <- data2017c[, -highCorr_C]
data2018c <- data2018c[, -highCorr_C]



# define model controls
ctrl_C <- trainControl(method = "LGOCV", summaryFunction = twoClassSummary,
                       classProbs = TRUE)








##### Part 3: Forecasting US Stock Behavior via Classification Modeling:
#####         Creating 6.5 classification forecasting models which generate 
#####         objectively comparable predictions. 
#####         The 6.5 classification models I chose are: 
#####         Logit, PLS-DA, Elastic Net, an Artificial Neural Network,
#####         an Average (of several) Neural Networks, 
#####         Support Vector Machines, and KNN.
#####         
###           Important note: when our comments say that some prediction 
###           function predict() is comparing the predicted classification 
###           for some stock in 2015 (given a classification model trained 
###           on the data from 2014) vs the observed classification in 2015.
###           That just means the degree to which stocks that 
###           our model predicted would go up (in 2015 based on how 
###           it was trained & cross-validated on the 2014 data) actually
###           went up. And likewise, the degree to which stocks our model 
###           predicted would go down in 2015 actually did go down.


### Classification Forecasting Model #1: Logit
# set a random seed to ensure the replicability of our predictions
set.seed(100)          # use the same seed for every model
ftLogitC1 <- train(x = data2014c, y = class2014c, method = "glm",
                   metric = "ROC", preProcess = c("center", "scale"),
                   trControl = ctrl_C)
ftLogitC1
class(ftLogitC1)
summary(ftLogitC1)

# compare the expected classifications in 2015 to the observed classifications in 2015
LogitC1_predictions_for_2015 <- predict(ftLogitC1, data2015c)
LogitC1_predictions_for_2016 <- predict(ftLogitC1, data2016c)   # same same for 2016
LogitC1_predictions_for_2016 <- predict(ftLogitC1, data2017c)   # same same for 2017

# create a confusion matrix to show how well our Logit model's predictions fared
# by inspecting all of its classification model performance metrics
LogitC1_CFM <- confusionMatrix(data = LogitC1_predictions_for_2015, 
                               reference = class2015c, positive = "Increase")
LogitC1_CFM

LogitC1_prob2015 <- predict(ftLogitC1, data2015c, type = "prob")
LogitC1_prob2016 <- predict(ftLogitC1, data2016c, type = "prob")
LogitC1_prob2017 <- predict(ftLogitC1, data2018c, type = "prob")


# create an ROC curve for our Logit model and store it in an object
ROC_LogitC1 <- roc(response = class2015c, predictor = LogitC1_prob2015$Increase,
                   levels = rev(levels(class2015c)))
# plot that ROC curve
plot(ROC_LogitC1, col = "red", lwd = 3, 
     main = "ROC curve for our Logit fit on the 2014 stock data")

# calculate the area under the ROC curve
auc_C1 <- auc(ROC_LogitC1)
cat('Area under the ROC curve for our Logistic Regression: ', round(auc_C1, 4))

# assess the importance of the included regressors  
VarImp_LogitC1 <- varImp(ftLogitC1)
VarImp_LogitC1

# create a variable importance plot (for only the top 10 most important regressors)
plot(VarImp_LogitC1, top = 10, 
     main = 'Variable Importance Plot for my Logit Model fit on the 2014 stock data')




### Classification Forecasting Model #1.5: Logit after PCA
set.seed(100)   # use the same seed for every model
data2014c_OV <- cbind(data2014c, class2014c)   # OV stands for original version
PCA_C1.fit = pcr(class2014c ~ ., data = data2014c_OV, 
                 scale = TRUE, validation = "CV")
PCA_C1.fit = pcr(formula = class2014c ~ ., data = data2014c_OV, 
                 scale = TRUE, validation = "CV")
PCA_C1.fit = prcomp(data2014c, center = TRUE, scale = TRUE)
head(PCA_C1.fit$x[, 1:10])







### Classification Model #2: Partial Least Squares Discriminant Analysis
set.seed(100)   # use the same seed for every model
ftPLSDA_C1 <- train(x = data2014c, y = class2014c, method = "pls",
                  tuneGrid = expand.grid(.ncomp = 1:5),
                  preProc = c("center", "scale"), metric = "ROC",
                  trControl = ctrl_C)
ftPLSDA_C1
summary(ftPLSDA_C1)

set.seed(100)
PLSDA_C1fits2 <- plsda(x = data2014c, y = class2014c, 
                       scale = TRUE, ncomp = 5)
PLSDA_C1fits2

# compare the expected classifications for each stock in 2015 to
# the observed classifications in 2015
PLSDA_C1predictions <- predict(ftPLSDA_C1, data2015c)
head(PLSDA_C1predictions)

PLSDA_C1probs <- predict(ftPLSDA_C1, newdata = data2015c, type = "prob")
head(PLSDA_C1probs)

# create a confusion matrix to show how well our PLS-DA model's predictions fared
# by inspecting all of its classification model performance metrics
PLSDA_C1_CFM <- confusionMatrix(data = PLSDA_C1predictions, 
                                reference = class2015c, positive = "Increase")
PLSDA_C1_CFM

PLSDA_C1prob2016 <- predict(ftPLSDA_C1, data2016c, type = "prob")
PLSDA_C1prob2017 <- predict(ftPLSDA_C1, data2017c, type = "prob")

# create an ROC curve for our PLS-DA model and store it in an object
rocPLSDA_C1 <- roc(response = class2015c, predictor = PLSDA_C1probs$Increase,
                   levels = rev(levels(class2015c)))

# show the plot of that ROC curve for our PLS regression
plot(rocPLSDA_C1, col = "red", lwd = 3, 
     main = "ROC curve for our PLS-DA model for the 2014 stock data")

auc_C2 <- auc(rocPLSDA_C1)
cat('Area under the ROC curve for our PLS-DA model: ', round(auc_C2, 4))

# assess the importance of the included regressors  
VarImp_PLSDA_C1 <- varImp(ftPLS_C1)
VarImp_PLSDA_C1

# create a variable importance plot (for only the top 20 IVs)
plot(VarImp_PLSDA_C1, top = 20, 
     main = 'Variable Importance Plot for our PLS-DA model fit on the 2014 data')







### Classification Forecasting Model #3: penalized regression models
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1), 
                        .lambda = seq(.01, .2, length = 40))

set.seed(100)        # use the same seed for every predictive model
# method 1 of fitting the elastic net via the train() function 
glmnC1_Tuned <- train(x = data2014c, y = class2014c, method = "glmnet",
                      tuneGrid = glmnGrid, preProc = c("center", "scale"),
                      metric = "ROC", trControl = ctrl_C)
glmnC1_Tuned

glmnC1_Predict2015 <- predict(glmnC1_Tuned, data2015c)
glmnC1_Predict2016 <- predict(glmnC1_Tuned, data2016c)
glmnC1_Predict2017 <- predict(glmnC1_Tuned, data2017c)
glmnC1_Predict2018 <- predict(glmnC1_Tuned, data2018c)


# method 2 of fitting the elastic net via the cv.glmnet() function 


# assess the performance of our penalized classification model which was
# trained on the 2014 data's predictions for 2015 via confusion matrix
PRMC1_CFM1 <- confusionMatrix(data = glmnPredictC1_2015, 
                              reference = class2015c, positive = "Increase")
PRMC1_CFM1 

# do the same for our model trained on the 2014 data's
# predictions for 2016
PRMC1_CFM2 <- confusionMatrix(data = glmnPredictC1_2016, 
                              reference = class2016c, positive = "Increase")
PRMC1_CFM2
# do the same for our model trained on the 2014 data's
# predictions for 2017
PRMC1_CFM3 <- confusionMatrix(data = glmnPredictC1_2017, 
                              reference = class2017c, positive = "Increase")
PRMC1_CFM3

glmnC1_Prob <- predict(glmnC1_Tuned, data2015c, type = "prob")

# create an ROC curve for our penalized classification model
glmnC1_ROC <- roc(response = class2015c, predictor = glmnC1_Prob$Increase,
                  levels = rev(levels(class2015c)))

# plot the ROC curve
plot(glmnC1_ROC, col = "red", lwd = 3, main = "ROC curve for our penalized model")

auc_C3 <- auc(glmnC1_ROC)
cat('Area under the ROC curve for our penalized model: ', 
    round(auc_C3, 4))

# assess the importance of the included regressors  
impGLMN_C1 <- varImp(glmnC1_Tuned)
impGLMN_C1

# create a variable importance plot (for only the top 20 regressors)
plot(impGLMN, top = 10, main = 'Variable Importance Plot for our penalized model')







### Classification Forecasting Model #4: Artificial Neural Network
nnGrid = expand.grid(.decay = c(0, 0.01, 0.1), .size = 1:10)

set.seed(100)
nnetModelC1 = train(x = data2014c, y = class2014c, method = "nnet", 
                    preProc = c("center", "scale"), linout = FALSE, 
                    trace = FALSE, MaxNWts = 10*(ncol(data2014c) + 1) + 10 + 1, 
                    maxit = 500, tuneGrid = nnGrid)
nnetModelC1

# compare the expected classifications in 2015 to the observed classifications in 2015
nnetC1Pred = predict(nnetModelC1, newdata = data2015c)

nnetPRc1 = postResample(pred = nnetC1Pred, obs = class2015c)
nnetPRc1

# assess the performance of our Neural Network's predictions for the 
# 2015 test set via a confusion matrix
nnet_CFM_C1 <- confusionMatrix(data = nnetPredC1, reference = class2015c,
                               positive = "Increase")
nnet_CFM_C1

nnetC1_Prob <- predict(nnetModelC1, data2015c, type = "prob")

nnetC1_ROC <- roc(response = class2015c, predictor = nnetC1_Prob$Increase,
                  levels = rev(levels(class2015c)))

plot(nnetC1_ROC, col = "red", lwd = 3, 
     main = "ROC curve for our Neural Network")

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
avNNetModelC1 = train(x = data2014c, y = class2014c, method = "avNNet", 
                    preProc = c("center", "scale"), linout = FALSE, 
                    trace = FALSE, MaxNWts = 10*(ncol(data2014c) + 1) + 10 + 1, 
                    maxit = 500)
avNNetModelC1

# compare the expected classifications in 2015 to the observed 
# classifications in 2015
avNNetC1_Pred = predict(avNNetModelC1, newdata = data2015c)

avNNetC1_PR = postResample(pred = avNNetC1_Pred, obs = class2015c)
avNNetC1_PR

# assess the performance of the predictions made by our 
# average neural nets model for the 2015 testing dataset via confusion matrix
avgNNc1_CFM <- confusionMatrix(data = avNNetC1_Pred, reference = class2015c,
                               positive = "Increase")
avgNNc1_CFM

avNNetC1_Prob <- predict(avNNetModelC1, data2015c, type = "prob")

avNNetModelC1_ROC <- roc(response = class2015c, 
                         predictor = avNNetC1_Prob$Increase,
                         levels = rev(levels(class2015c)))

plot(avNNetModelC1_ROC, col = "red", lwd = 3, 
     main = "ROC curve for our Average Neural Net")
# calculate the area under the above ROC curve
auc_C5 <- auc(avNNetModelC1_ROC)
cat('Area under the ROC curve for our Average Neural Net:', round(auc_C5, 4))

# assess the importance of the included regressors  
impAVNNet_c1 <- varImp(avNNetModelC1)
impAVNNet_c1

# create a variable importance plot
plot(impAVNNet_c1, top = 10, 
     main = 'Variable Importance for Average Neural Network')







### Classification Forecasting Model #5: Support Vector Machine
library(kernlab)
set.seed(100)
svmGrid = expand.grid(.sigma = c(0, 0.01, 0.1), .C = 1:10)

svmC1Fit <- train(x = data2014c, y = class2014c,  method = "svmRadial",
                  preProc = c("center", "scale"), tuneGrid = svmGrid,
                  trControl = trainControl(method = "repeatedcv", 
                                           repeats = 5, classProbs =  TRUE))
svmC1Fit

# compare the expected classifications in 2015 to the observed classifications in 2015
svmC1_Pred = predict(svmC1Fit, newdata = data2015c)

svmC1_PR = postResample(pred = svmC1_Pred, obs = class2015c)
svmC1_PR

# assess the performance of our penalized classification model's predictions 
# via a confusion matrix
svmC1_CFM <- confusionMatrix(data = svmC1_Pred, reference = class2015c,
                             positive = "Increase")
svmC1_CFM

svmC1Prob <- predict(svmC1Fit, data2015c, type = "prob")

svmC1_ROC <- roc(response = class2015c, predictor = svmC1Prob$Increase,
                 levels = rev(levels(class2015c)))

plot(svmC1_ROC, col = "red", lwd = 3, main = "ROC curve for our SVM model")

# calculate the area under the above ROC curve
auc_C7 <- auc(svmC1_ROC)
cat('Area under the ROC curve for our Support Vector Machine model: ', 
    round(auc_C7, 4))







### Classification Forecasting Model #6: K-Nearest Neighbors 
set.seed(100)
knnC1_Model = train(x = data2014c, y = class2014c, method = "knn",
                    preProc = c("center","scale"), tuneLength = 20)
knnC1_Model

# compare the expected classifications in 2015 to the observed classifications in 2015
knnC1_Pred = predict(knnC1_Model, newdata = data2015c)

knnC1_PR = postResample(pred = knnC1_Pred, obs = class2015c)
knnC1_PR

knnC1_CFM <- confusionMatrix(data = knnC1_Pred, reference = class2015c,
                             positive = "Increase")
knnC1_CFM

knnC1_Prob <- predict(knnC1_Model, data2015c, type = "prob")

knnC1_ROC <- roc(response = class2015c, predictor = knnC1_Prob$Increase,
                 levels = rev(levels(class2015c)))

plot(knnROC, col = "red", lwd = 3, main = "ROC curve for our KNN Model")

# calculate the area under the above ROC curve
auc_C8 <- auc(knnC1_ROC)
cat('Area under the ROC curve for our KNN model fit on the 2014 data: ', 
    round(auc_C8, 4))





#### Part 4: Classification Model Performance Assessment & Comparison
####         print out the Confusion Matrices for all 8 of our 
####         classification models again, but this time, print them 
####         out right on top of each other, one after the other in
####         order to facilitate easy visual comparisons of each of their
####         classification performance metrics with all the other 7.
LogitC1_CFM         # the confusion matrix for our logistic regression model

PLSDA_C1_CFM        # the confusion matrix for our PLS-DA model

PRMC1_CFM1          # the confusion matrix for our penalized regression model

nnet_CFM_C1         # the confusion matrix for our Neural Network model

avgNNc1_CFM         # the confusion matrix for our average neural net

svmC1_CFM           # the confusion matrix for our Support Vector Machine model

knnC1_CFM           # the confusion matrix for our K-Nearest Neighbors model
