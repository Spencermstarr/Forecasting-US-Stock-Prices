##### Group 4's Collective Project Code
### Part 1: setting up the environment and loading the necessary packages
library(caret)
library(earth)
library(vip)
library(plyr)
library(pROC)
library(nnet)
library(parallel)
library(doParallel)

#allow multithreaded operating
threads <- detectCores()
cluster <- makePSOCKcluster(threads)
registerDoParallel(cluster)


# find out which working directory R has defaulted to
getwd()
# use Ctrl+Shift+H to select the proper WD if necessary, for this script,
# the proper working directory will be the unzipped version of the zip folder
# we turned everything required of our final deliverable for this project in with

## load the training data and the testing data into R
data2014 <- read.csv("2014_Financial_Data.csv", header = TRUE)
data2015 <- read.csv("2015_Financial_Data.csv", header = TRUE)

count(data2014, "Class")
count(data2015, "Class")




### Part 2: data cleaning and munging

# remove all predictors with near zero variance
ZeroVar <- nearZeroVar(data2014)

data2014 <- data2014[ , -ZeroVar]
data2015 <- data2015[ , -ZeroVar]


# remove all rows with missing values from the 2014 financial dataset
mean(is.na(data2014))
data2014 <- na.omit(data2014)
mean(is.na(data2014))

# remove all rows with missing values from the 2015 financial dataset
mean(is.na(data2015))
data2015 <- na.omit(data2015)
mean(is.na(data2015))

count(data2014, "Class")
count(data2015, "Class")

# remove all non-numeric columns
x2014 <- data2014$X
class2014 <- data2014$Class
sector2014 <- data2014$Sector

x2015 <- data2015$X
class2015 <- data2015$Class
sector2015 <- data2015$Sector

class2014 <- ifelse(class2014 == 1, "Increase", "Decrease")
class2015 <- ifelse(class2015 == 1, "Increase", "Decrease")


# convert the integer values in the Class column of both yearly stock market 
# datasets (which are both stored in their own dataframe) into factors stored in
# their own newly created objects separate from the dataframes they came from
class2014 <- as.factor(class2014)
class2015 <- as.factor(class2015)

# remove the Class and Sector columns from both annual dataframes
data2014 <- subset(data2014, select = -c(X, Class, Sector))
data2015 <- subset(data2015, select = -c(X, Class, Sector))



# find and remove highly correlated predictors
correlations <- cor(data2014)

highCorr <- findCorrelation(correlations, cutoff = .8)
length(highCorr)
data2014 <- data2014[, -highCorr]
data2015 <- data2015[, -highCorr]


# remove stock price variance column to prevent perfect multicollinearity
data2014 <- subset(data2014, select = -c(X2015.PRICE.VAR....))
data2015 <- subset(data2015, select = -c(X2016.PRICE.VAR....))


# define our model controls
ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)




### Part 3: creating several financial forecasting models which 
###         generate objectively comparable predictions. 
###         Important note: when our comments say that some predict() is comparing
###         the expected classification in 2015 vs the observed classification in 
###         2015, that just means the degree to which stocks our model predicted would go up
###         (in 2015 based on how it was trained & cross-validated on the 2014 data)
###         actually went up. And likewise, the degree to which stocks our model 
###         predicted would go down in 2015 actually did go down.

## Forecasting Model #1: Logit
# set a seed to ensure the replicability of our predictions
set.seed(100)          # use the same seed for every model
ftLogit <- train(x = data2014,
                 y = class2014,
                 method = "glm",
                 metric = "ROC",
                 preProcess = c("center", "scale"),
                 trControl = ctrl)
ftLogit

# compare the expected classifications in 2015 to the observed classifications in 2015
Logitpredict <- predict(ftLogit, data2015)


# create a confusion matrix to show how well our Logit model's predictions fared
# by inspecting all of its classification model performance metrics
Logit_CFM <- confusionMatrix(data = Logitpredict,  
                reference = class2015,
                positive = "Increase")
Logit_CFM


Logit_prob <- predict(ftLogit, data2015, type = "prob")

# create an ROC curve for our Logit model and store it in an object
ROC_Logit <- roc(response = class2015, predictor = Logit_prob$Increase,
              levels = rev(levels(class2015)))

# plot that ROC curve
plot(ROC_Logit, col = "red", lwd = 3, main = "ROC curve for our Logit ")

# calculate the area under the above ROC curve
auc1 <- auc(ROC_Logit)
cat('Area under the ROC curve for our Logistic Regression: ', round(auc1, 4))


# assess the importance of the included regressors  
impLogit <- varImp(ftLogit)
impLogit

# create a variable importance plot (for only the top regressors)
plot(impLogit, top = 10, main = 'Variable Importance Plot for our Logit Model')








## Forecasting Model #2: Partial Least Squares Discriminant Analysis
set.seed(100)   # use the same seed for every model

ftPLS <- train(x = data2014, y = class2014, method = "pls",
               tuneGrid = expand.grid(.ncomp = 1:4),
               preProc = c("center", "scale"), metric = "ROC",
               trControl = ctrl)
ftPLS

# compare the expected classifications in 2015 to the observed classifications in 2015
PLSpredict <- predict(ftPLS, data2015)

# create a confusion matrix to show how well our PLS-DA model's predictions fared
# by inspecting all of its classification model performance metrics
PLSDA_CFM <- confusionMatrix(data = PLSpredict, 
                reference = class2015,
                positive = "Increase")
PLSDA_CFM

PLSprob <- predict(ftPLS, data2015, type = "prob")

# create an ROC curve for our PLS-DA model and store it in an object
rocPLS <- roc(response = class2015, predictor = PLSprob$Increase,
              levels = rev(levels(class2015)))

# show the plot of that ROC curve for our PLS regression
plot(rocPLS, col = "red", lwd = 3, main = "ROC curve for our PLS-DA model")

auc2 <- auc(rocPLS)
cat('Area under the ROC curve for our PLS-DA model: ', round(auc2, 4))

# assess the importance of the included regressors  
impPLS <- varImp(ftPLS)
impPLS

# create a variable importance plot (for only the top 20 IVs)
plot(impPLS, top = 10, main = 'Variable Importance Plot for our PLS-DA model')







## Forecasting Model #3: penalized regression model
set.seed(100)        # use the same seed for every predictive model

glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 40))

glmnTuned <- train(x = data2014, y = class2014, method = "glmnet",
                   tuneGrid = glmnGrid, preProc = c ("center", "scale"),
                   metric = "ROC", trControl = ctrl)
glmnTuned

glmnPredict <- predict(glmnTuned, data2015)


# assess the performance of our penalized classification model's predictions 
# via a confusion matrix
PRM_CFM <- confusionMatrix(data = glmnPredict, 
                reference = class2015,
                positive = "Increase")
PRM_CFM

glmnProb <- predict(glmnTuned, data2015, type = "prob")

# create an ROC curve for our penalized classification model
glmnROC <- roc(response = class2015, predictor = glmnProb$Increase,
              levels = rev(levels(class2015)))

# plot the ROC curve
plot(glmnROC, col = "red", lwd = 3, main = "ROC curve for our penalized model")

auc3 <- auc(glmnROC)
cat('Area under the ROC curve for our elastic net: ', 
    round(auc3, 4))

# assess the importance of the included regressors  
impGLMN <- varImp(glmnTuned)
impGLMN

# create a variable importance plot (for only the top 20 regressors)
plot(impGLMN, top = 10, main = 'Variable Importance Plot for our penalized model')







## Forecasting Model #4: Neural Network
nnGrid = expand.grid(.decay = c(0, 0.01, 0.1), .size = 1:10)

set.seed(100)

nnetModel = train(x = data2014, y = class2014, method = "nnet", 
                  preProc = c("center", "scale"), linout = FALSE, 
                  trace = FALSE, MaxNWts = 10 * (ncol(data2014)+1) + 10 + 1, 
                  maxit = 500, tuneGrid = nnGrid)
nnetModel

# compare the expected classifications in 2015 to the observed classifications in 2015
nnetPred = predict(nnetModel, newdata = data2015)

nnetPR = postResample(pred = nnetPred, obs = class2015)
nnetPR

# assess the performance of our Neural Network's predictions for the 2015 test set
# via a confusion matrix
nnet_CFM <- confusionMatrix(data = nnetPred, 
                reference = class2015,
                positive = "Increase")
nnet_CFM

nnetProb <- predict(nnetModel, data2015, type = "prob")

nnetROC <- roc(response = class2015, predictor = nnetProb$Increase,
               levels = rev(levels(class2015)))

plot(nnetROC, col = "red", lwd = 3, main = "ROC curve for our Neural Network")

# calculate the area under the above ROC curve
auc4 <- auc(nnetROC)
cat('Area under the ROC curve for our Neural Network:', round(auc4, 4))

# assess the importance of the included regressors  
impNNET <- varImp(nnetModel)
impNNET

# create a variable importance plot
plot(impNNET, top = 10, main = 'Variable Importance for Neural Network')













## Forecasting Model #5: MARS
marsGrid = expand.grid(.degree = 1:2, .nprune = 2:38)
set.seed(100)
marsModel = train(x = data2014, y = class2014, method = "earth", 
                  preProc = c("center", "scale"), tuneGrid = marsGrid)
marsModel

# compare the expected classifications in 2015 to the observed classifications in 2015
marsPred = predict(marsModel, newdata = data2015)

marsPR = postResample(pred = marsPred, obs = class2015)
marsPR

# assess the prediction performance metrics for our Multivariate Adaptive 
# Regression Splines model via a confusion matrix
mars_CFM <- confusionMatrix(data = marsPred, 
                reference = class2015,
                positive = "Increase")
mars_CFM

marsProb <- predict(marsModel, data2015, type = "prob")

marsROC <- roc(response = class2015, predictor = marsProb$Increase,
                 levels = rev(levels(class2015)))

plot(marsROC, col = "red", lwd = 3, main = "ROC curve for our MARS model")

# calculate the area under the above ROC curve
auc5 <- auc(marsROC)
cat('Area under the ROC curve for our MARS model: ', round(auc5, 4))






## Forecasting Model #6: SVM
set.seed(100)
svmGrid = expand.grid(.sigma = c(0, 0.01, 0.1), .C = 1:10)

svmFit <- train(x = data2014, y = class2014,  method = "svmRadial",
                preProc = c("center", "scale"), tuneGrid = svmGrid,
                trControl = trainControl(method = "repeatedcv", repeats = 5,
                                         classProbs =  TRUE))
svmFit

# compare the expected classifications in 2015 to the observed classifications in 2015
svmPred = predict(svmFit, newdata = data2015)

svmPR = postResample(pred = svmPred, obs = class2015)
svmPR

# assess the performance of our penalized classification model's predictions 
# via a confusion matrix
svm_CFM <- confusionMatrix(data = svmPred, 
                reference = class2015,
                positive = "Increase")
svm_CFM

svmProb <- predict(svmFit, data2015, type = "prob")

svmROC <- roc(response = class2015, predictor = svmProb$Increase,
               levels = rev(levels(class2015)))

plot(svmROC, col = "red", lwd = 3, main = "ROC curve for our SVM model")

# calculate the area under the above ROC curve
auc6 <- auc(svmROC)
cat('Area under the ROC curve for our Support Vector Machine model: ', 
    round(auc6, 4))







## Forecasting Model #7: K-Nearest Neighbors 
set.seed(100)
knnModel = train(x = data2014, y = class2014, method = "knn",
                 preProc = c("center","scale"), tuneLength = 20)
knnModel

# compare the expected classifications in 2015 to the observed classifications in 2015
knnPred = predict(knnModel, newdata = data2015)

knnPR = postResample(pred = knnPred, obs = class2015)
knnPR


knn_CFM <- confusionMatrix(data = knnPred, 
                reference = class2015,
                positive = "Increase")
knn_CFM

knnProb <- predict(knnModel, data2015, type = "prob")

knnROC <- roc(response = class2015, predictor = knnProb$Increase,
              levels = rev(levels(class2015)))

plot(knnROC, col = "red", lwd = 3, main = "ROC curve for our KNN Model")

# calculate the area under the above ROC curve
auc7 <- auc(knnROC)
cat('Area under the ROC curve for our KNN model: ', round(auc7, 4))











## Ensemble Learning Model #1: average neural network
set.seed(100)
avNNetModel = train(x = data2014, y = class2014, method = "avNNet", 
                    preProc = c("center", "scale"), linout = FALSE,
                    trace = FALSE, MaxNWts = 10 * (ncol(data2014)+1) + 10 + 1, 
                    maxit = 500)
avNNetModel

# compare the expected classifications in 2015 to the observed classifications in 2015
avNNetPred = predict(avNNetModel, newdata = data2015)

avNNetPR = postResample(pred = avNNetPred, obs = class2015)
avNNetPR

# assess the performance of the predictions made by our average neural nets model
# for the 2015 testing dataset via a confusion matrix
avgNN_CFM <- confusionMatrix(data = avNNetPred, 
                             reference = class2015,
                             positive = "Increase")

avNNetProb <- predict(avNNetModel, data2015, type = "prob")

avNNetROC <- roc(response = class2015, predictor = avNNetProb$Increase,
                 levels = rev(levels(class2015)))

plot(avNNetROC, col = "red", lwd = 3, 
     main = "ROC curve for our Average Neural Net")

# calculate the area under the above ROC curve
ensemble_auc1 <- auc(nnetROC)
cat('Area under the ROC curve for our Average Neural Net:', 
    round(ensemble_auc1, 4))

# assess the importance of the included regressors  
impAVNNet <- varImp(avNNetModel)
impAVNNet

# create a variable importance plot
plot(impAVNNet, top = 10, main = 'Variable Importance for Average Neural Network')











## Ensemble Learning Model #2: Random Forest
set.seed(100)  # use the same seed for every model
# Define the Tuning Grid
rfGrid <- expand.grid(.mtry = c(1:sqrt(ncol(data2014))))  # sqrt of total number of variables is a common choice

# Train the Random Forest Model using the caret package
ftRF <- train(x = data2014, 
              y = class2014, 
              method = "rf", 
              tuneGrid = rfGrid, 
              metric = "ROC", 
              trControl = ctrl)
# model summary
ftRF

# use the model fitted on the 2014 data to predict the 2015 data
RFpredict <- predict(ftRF, newdata = data2015)

# performance assessment
RF_CFM <- confusionMatrix(data = RFpredict, 
                          reference = class2015, 
                          positive = "Increase")
RF_CFM

# construct the ROC Curve and then calculate the Area Under the Curve (AUC)
RFprob <- predict(ftRF, newdata = data2015, type = "prob")
ROC_RF <- roc(response = class2015, predictor = RFprob$Increase,
              levels = rev(levels(class2015)))

plot(ROC_RF, col = "red", lwd = 3, main = "ROC curve for our Random Forest Model")

ensemble_auc2 <- auc(ROC_RF)
cat('Area under the ROC curve for our Random Forest model: ', 
    round(ensemble_auc2, 4))

# variable importance evaluation
impRF <- varImp(ftRF)
plot(impRF, top = 10, main = 'Variable Importance Plot for our Random Forest Model')








## Ensemble Learning Model #3: Meta-Learning Model
# First, we create a caretList containing all the individual models we want to ensemble.
models_list <- caretList(x = data2014, y = class2014,
                         trControl = ctrl, metric = "ROC",
                         methodList = c("glm", "pls", "glmnet", "rf"))

# predict the 2015 stock behavior
Logit_prob <- predict(ftLogit, newdata = data2015, type = "prob")[, "Increase"]
PLS_prob <- predict(ftPLS, newdata = data2015, type = "prob")[, "Increase"]
glmn_prob <- predict(glmnTuned, newdata = data2015, type = "prob")[, "Increase"]
RF_prob <- predict(ftRF, newdata = data2015, type = "prob")[, "Increase"]

ensemble_data_2015 <- data.frame(Logit_prob, PLS_prob, glmn_prob, RF_prob)

# train this ensemble on the 2014 data
# Install and load the caretEnsemble package if not already installed
if (!requireNamespace("caretEnsemble", quietly = TRUE)) {
  install.packages("caretEnsemble") }
library(caretEnsemble)


# Create the Meta Model
set.seed(100)  # for reproducibility
meta_model <- train(ensemble_data, class2015, method = "glm",
                    metric = "ROC", trControl = trainControl(method = "boot",
                                             number = 50,
                                             summaryFunction = twoClassSummary,
                                             classProbs = TRUE))
meta_model


## model performance assessment
#meta_predictions_2015 <- predict(meta_model, newdata = ensemble_data_2015)
meta_model_prob_2015 <- predict(meta_model, newdata = ensemble_data_2015, 
                                type = "prob")[, "Increase"]

# create a confusion matrix
Meta_Model_CFM <- confusionMatrix(data = meta_model_prob_2015, 
                                reference = class2015, positive = "Increase")
Meta_Model_CFM

# Create the ROC curve
ROC_ensemble <- roc(response = class2015, predictor = Ensemble_prob_2015,
                    levels = rev(levels(class2015)))

plot(ROC_ensemble, col = "red", lwd = 3, 
     main = "ROC curve for our Ensemble Learning Model")

ensemble_auc3 <- auc(ROC_ensemble)
cat('Area under the ROC curve for our Meta Learning Model: ', 
    round(ensemble_auc3, 4))

# variable importance evaluation
## NOTE: I did this the wrong way, but the result of it is 
## kind of interesting
imp_ensemble <- varImp(meta_model)
plot(imp_ensemble, top = 10, 
     main = 'Variable Importance Plot for our Ensemble Learning Model')















### Part 4: model comparison and assessment

# print out the Confusion Matrices and AUCs for all 7 of the individual classification 
# algorithms again, then do the same for the 3 ensemble learning models.
# But this time, print them out right on top of each other, one after the other
# in order to facilitate easy visual comparison of each of their 
# classification performance metrics with each other. 
Logit_CFM          # the confusion matrix for our logistic regression model
cat('Area under the ROC curve for our Logistic Regression: ', round(auc1, 4))

PLSDA_CFM          # the confusion matrix for our PLS-DA model
cat('Area under the ROC curve for our PLS-DA model: ', round(auc2, 4))

PRM_CFM            # the confusion matrix for our penalized regression model
cat('Area under the ROC curve for our elastic net: ', round(auc3, 4))

nnet_CFM           # the confusion matrix for our Artificial Neural Network model
cat('Area under the ROC curve for our Neural Network:', round(auc4, 4))

mars_CFM           # the confusion matrix for our MARS model
cat('Area under the ROC curve for our MARS model: ', round(auc5, 4))

svm_CFM            # the confusion matrix for our Support Vector Machine model
cat('Area under the ROC curve for our Support Vector Machine model: ', round(auc6, 4))

knn_CFM            # the confusion matrix for our K-Nearest Neighbors model
cat('Area under the ROC curve for our KNN model: ', round(auc7, 4))




avgNN_CFM          # the confusion matrix for our average neural net
cat('Area under the ROC curve for our Average Neural Net:', 
    round(ensemble_auc1, 4))

RF_CFM             # the confusion matrix for our Random Forest model
cat('Area under the ROC curve for our Random Forest model: ', 
    round(ensemble_auc2, 4))

Meta_Model_CFM     # the confusion matrix for our Meta Learning model
cat('Area under the ROC curve for our Meta Learning Model: ', 
    round(ensemble_auc3, 4))

