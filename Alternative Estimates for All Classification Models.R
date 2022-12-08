##### Alternative estimates, i.e. alternately fitted predictive models
##### of the same types, just trained on the 2015, 2016, or 2017 data instead.

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




### interpolate the missing values in data2014 using their medians
# find and count the position of all NAs (column-wise)
which(is.na(data2014))
# find all rows with at least 1 NA
which(rowSums(is.na(data2014)) != 0)
# count all NAs in data2014
sum(is.na(data2014))
# find/count the # of NAs in each column of data2014
colSums(is.na(data2014))

median_data2014 <- lapply(na.omit(data2014), median)
# mean_data2014 <- lapply(mean_data2014, round)
for(i in 1:length(data2014)) {
  data2014m[is.na(data2014[, i]), i] <- median(data2014[, i], na.rm = TRUE) }
# count all NAs in data2014
sum(is.na(data2014m))
# now remove the mean_data2014 list since its purpose has been served
rm(median_data2014)

# interpolate the missing values in data2015
median_data2015 <- lapply(na.omit(data2015), median)
for(i in 1:length(data2015)) {
  data2015m[is.na(data2015[, i]), i] <- median(data2015[, i], na.rm = TRUE) }
rm(median_data2015)

# interpolate the missing values in data2016
median_data2016 <- lapply(na.omit(data2016), median)
for(i in 1:length(data2016)) {
  data2016m[is.na(data2016[, i]), i] <- median(data2016[, i], na.rm = TRUE) }
rm(median_data2016)

# interpolate the missing values in data2017
median_data2017 <- lapply(na.omit(data2017), median)
for(i in 1:length(data2017)) {
  data2017m[is.na(data2017[, i]), i] <- median(data2017[, i], na.rm = TRUE) }
rm(median_data2017)

# interpolate the missing values in data2018
sum(is.na(data2018))
median_data2018 <- lapply(na.omit(data2018), median)
for(i in 1:length(data2018)) {
  data2018m[is.na(data2018[, i]), i] <- median(data2018[, i], na.rm = TRUE) }
# count all NAs in data2018
sum(is.na(data2018))
# remove the mean_data2014 list since its purpose has already been served
rm(mean_data2018)





# define model controls
ctrl_c <- trainControl(method = "LGOCV", summaryFunction = twoClassSummary,
                       classProbs = TRUE)









### Classification Forecasting Model #1: Logit
# set a random seed to ensure the replicability of our predictions
set.seed(100)          # use the same seed for every model
ft.LogitC1_V2 <- train(x = data2014, y = class2014, method = "glm",
                   metric = "ROC", preProcess = c("center", "scale"),
                   trControl = ctrl_c)
ft.LogitC1_V2

# compare the expected classifications in 2015 to the observed classifications in 2015
LogitC1_V2preds_for_2015 <- predict(ft.LogitC1_V2, data2015)
LogitC1_predictions_for_2016 <- predict(ftLogitC1, data2016)   # same same for 2016
LogitC1_predictions_for_2016 <- predict(ftLogitC1, data2017)   # same same for 2017

# create a confusion matrix to show how well our Logit model's predictions fared
# by inspecting all of its classification model performance metrics
LogitC1_v2CFM <- confusionMatrix(data = LogitC1_V2preds_for_2015, 
                               reference = class2015, positive = "Increase")
LogitC1_v2CFM

LogitC1_V2prob2015 <- predict(ft.LogitC1_V2, data2015, type = "prob")

# create an ROC curve for our Logit model and store it in an object
ROC_LogitC1_V2 <- roc(response = class2015, 
                      predictor = LogitC1_V2prob2015$Increase,
                      levels = rev(levels(class2015)))
# plot that ROC curve
plot(ROC_LogitC1_V2, col = "red", lwd = 3, 
     main = "ROC curve for our Logit fit on the 2014 stock data")

# calculate the area under the ROC curve
auc_LogitV2 <- auc(ROC_LogitC1_V2)
cat('Area under the ROC curve for Logistic Regression: ', 
    round(auc_LogitV2, 3))

# assess the importance of the included regressors  
VarImp_LogitC1 <- varImp(ftLogitC1)
VarImp_LogitC1

# create a variable importance plot (for only the top 10 most important regressors)
plot(VarImp_LogitC1, top = 10, 
     main = 'Variable Importance Plot for my Logit Model fit on the 2014 stock data')



# try fitting the Logit using the glm() function instead
LogitC2.fits = glm(formula = class2014c ~ ., family = binomial, 
                   data = data2014c)
LogitC2.fits = glm(formula = class2014c ~ ., family = binomial, 
                   data = data2014c_OV)
LogitC2.fits

null_logitC2 <- glm(formula = class2014c ~ 1, family = binomial,
                data = data2014)
null_logitC2
summary(null_logitC2)



# compare the expected classifications in 2016 as predicted by the Logit
# model fit on the 2014 stock data to the observed classifications in 2016
LogitC1_predictions_for_2016 <- predict(ftLogitC1, data2016c)
# same same for 2017
LogitC1_predictions_for_2017 <- predict(ftLogitC1, data2017c)
LogitC1_predictions_for_2018 <- predict(ftLogitC1, data2018c)   #same same for 2018


# create a confusion matrix to show how well our Logit model trained
# on the 2014 data's predictions for the 2016 dataset fared 
# by inspecting all of its classification model performance metrics
LogitC1_CFM2 <- confusionMatrix(data = LogitC1_predictions_for_2016, 
                               reference = class2016c, positive = "Increase")
LogitC1_CFM2
# create a confusion matrix to show how well our Logit model trained
# on the 2014 data's predictions for the 2017 dataset fared 
# by inspecting all of its classification model performance metrics
LogitC1_CFM3 <- confusionMatrix(data = LogitC1_predictions_for_2017, 
                                reference = class2017c, positive = "Increase")
LogitC1_CFM3
# create a confusion matrix to show how well our Logit model trained
# on the 2014 data's predictions for the 2016 dataset fared 
# by inspecting all of its classification model performance metrics
LogitC1_CFM4 <- confusionMatrix(data = LogitC1_predictions_for_2018, 
                                reference = class2018c, positive = "Increase")
LogitC1_CFM4

LogitC1_prob2016 <- predict(ftLogitC1, data2016c, type = "prob")
LogitC1_prob2017 <- predict(ftLogitC1, data2017c, type = "prob")
LogitC1_prob2018 <- predict(ftLogitC1, data2018c, type = "prob")












### Classification Forecasting Model #2: PLA-DA
set.seed(100)   # use the same seed for every model
fit.PLS_C1V2 <- train(x = data2014, y = class2014, method = "pls",
                  tuneGrid = expand.grid(.ncomp = 1:4),
                  preProc = c("center", "scale"), metric = "ROC",
                  trControl = ctrl_c)
fit.PLS_C1V2

# compare the expected classifications in 2015 to the observed classifications in 2015
PLS_C1V2_preds <- predict(fit.PLS_C1V2, data2015)

# create a confusion matrix to show how well our PLS-DA model's predictions fared
# by inspecting all of its classification model performance metrics
PLSDA_C1V2_CFM <- confusionMatrix(data = PLS_C1V2_preds, reference = class2015,
                                  positive = "Increase")
PLSDA_C1V2_CFM
PLSDA_C1V2prob2015 <- predict(fit.PLS_C1V2, data2015, type = "prob")

# create an ROC curve for our PLS-DA model and store it in an object
rocPLSDA_C1V2 <- roc(response = class2015, predictor = PLSDA_C1V2prob2015$Increase,
                   levels = rev(levels(class2015)))
# show the plot of that ROC curve for our PLS regression
plot(rocPLSDA_C1V2, col = "red", lwd = 3, 
     main = "ROC curve for PLS-DA fit on all of the 2014 stock data")
# calculate its AUC
auc_PLSDA_C1V2 <- auc(rocPLSDA_C1V2)
cat('Area under the ROC curve for our PLS-DA model: ', 
    round(auc_PLSDA_C1V2, 3))

# assess the importance of the included regressors  
VarImp_PLSDA_C1V2 <- varImp(ftPLS_C1)
VarImp_PLSDA_C1V2

# create a variable importance plot (for only the top 20 IVs)
plot(VarImp_PLSDA_C1V2, top = 10, 
     main = 'Variable Importance Plot for our PLS-DA model fit on the 2014 data')














### Classification Forecasting Model #3: penalized regression models
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1), 
                        .lambda = seq(.01, .2, length = 40))

set.seed(100)        # use the same seed for every predictive model
glmnV2.TunedC1 <- train(x = data2014, y = class2014, method = "glmnet",
                     tuneGrid = glmnGrid, preProc = c ("center", "scale"),
                     metric = "ROC", trControl = ctrl_c)
glmnV2.TunedC1

glmnC1V2_Preds <- predict(glmnV2.TunedC1, data2015)
#glmnPredictC1_2016 <- predict(glmnTunedC1, data2016c)
#glmnPredictC1_2017 <- predict(glmnTunedC1, data2017c)

# assess the performance of our penalized classification model's 
# predictions via confusion matrix
prmC1V2_CFM <- confusionMatrix(data = glmnC1V2_Preds, reference = class2015,
                               positive = "Increase")
prmC1V2_CFM

glmnV2Prob <- predict(glmnV2.TunedC1, data2015, type = "prob")

# create an ROC curve for our penalized classification model
glmnROC_C1V2 <- roc(response = class2015, predictor = glmnV2Prob$Increase,
                  levels = rev(levels(class2015)))
# plot the ROC curve
plot(glmnROC_C1V2, col = "red", lwd = 3, 
     main = "ROC curve for elastic net fit on the full 2014 stock data,
     the version with all of its missing values interpolated")

auc_glmnV2 <- auc(glmnROC_C1V2)
cat('Area under the ROC curve for our penalized model: ', 
    round(auc_glmnV2, 3))

# assess the importance of the included regressors  
impGLMN <- varImp(glmnTunedC1)
impGLMN

# create a variable importance plot (for only the top 20 regressors)
plot(impGLMN, top = 10, main = 'Variable Importance Plot for our penalized model')


glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1), 
                        .lambda = seq(.01, .2, length = 40))

glmnC1_Predict2016 <- predict(glmnC1_Tuned, data2016c)
glmnC1_Predict2017 <- predict(glmnC1_Tuned, data2017c)
glmnC1_Predict2018 <- predict(glmnC1_Tuned, data2018c)


set.seed(100)        # use the same seed for every predictive model
glmnC2_Tuned <- train(x = data2015c, y = class2015c, method = "glmnet",
                      tuneGrid = glmnGrid, preProc = c ("center", "scale"),
                      metric = "ROC", trControl = ctrl_C)
glmnC2_Tuned
summary(glmnC2_Tuned, options(max.print = 100))
print(glmnC2_Tuned, options(max.print = 100))


set.seed(100)        # use the same seed for every predictive model
glmnC3_Tuned <- train(x = data2016c, y = class2016c, method = "glmnet",
                      tuneGrid = glmnGrid, preProc = c ("center", "scale"),
                      metric = "ROC", trControl = ctrl_C)
glmnC3_Tuned



# assess the performance of our penalized classification model which was
# trained on the 2014 data's predictions for 2016 via confusion matrix
PRMC1_CFM2 <- confusionMatrix(data = glmnPredictC1_2016, 
                              reference = class2016c, positive = "Increase")
PRMC1_CFM2
# do the same for our model trained on the 2014 data's
# predictions for 2017
PRMC1_CFM3 <- confusionMatrix(data = glmnPredictC1_2017, 
                              reference = class2017c, positive = "Increase")
PRMC1_CFM3
# do the same for our model trained on the 2014 data's
# predictions for 2017
PRMC1_CFM4 <- confusionMatrix(data = glmnPredictC1_2018, 
                              reference = class2018c, positive = "Increase")
PRMC1_CFM4











### Classification Forecasting Model #4: Artificial Neural Network
nnGrid = expand.grid(.decay = c(0, 0.01, 0.1), .size = 1:10)

set.seed(100)
nnetC1V2 = train(x = data2014, y = class2014, method = "nnet", 
                      preProc = c("center", "scale"), linout = FALSE, 
                      trace = FALSE, 
                      MaxNWts = 10 * (ncol(data2014) + 1) + 10 + 1, 
                      maxit = 500, tuneGrid = nnGrid)
nnetC1V2

# compare the expected classifications in 2015 to the observed classifications in 2015
nnetC1V2_Preds = predict(nnetC1V2, newdata = data2015)

nnetC1V2_PR = postResample(pred = nnetC1V2_Preds, obs = class2015)
nnetC1V2_PR

# assess the performance of our Neural Network's predictions for the 2015 test set
# via a confusion matrix
nnetC1V2_CFM <- confusionMatrix(data = nnetC1V2_Preds, reference = class2015,
                                positive = "Increase")
nnetC1V2_CFM

nnetC1V2_Prob <- predict(nnetC1V2, data2015, type = "prob")

nnetC1V2_ROC <- roc(response = class2015, predictor = nnetC1V2_Prob$Increase,
                    levels = rev(levels(class2015)))

plot(nnetC1V2_ROC, col = "red", lwd = 3, 
     main = "ROC curve for the Neural Network fit on the interpolated 
2014 stock data and tested on the interpolated 2015 data")

# calculate the area under the above ROC curve
auc_nnetC1V2 <- auc(nnetC1V2_ROC)
cat('Area under the ROC curve for our Neural Network:', 
    round(auc_nnetC1V2, 3))

# assess the importance of the included regressors  
impNNETc1 <- varImp(nnetModelC1)
impNNETc1

# create a variable importance plot
plot(impNNETc1, top = 10, main = 'Variable Importance for Neural Network')












## Classification Forecasting Model #4.5: Average Neural Network
set.seed(100)
avNNetModelC1V2 = train(x = data2014, y = class2014, method = "avNNet", 
                        preProc = c("center", "scale"), linout = FALSE, 
                        trace = FALSE, 
                        MaxNWts = 10 * (ncol(data2014) + 1) + 10 + 1, 
                        maxit = 500)
avNNetModelC1V2

# compare the expected classifications in 2015 to the observed classifications in 2015
avNNetC1V2_Preds = predict(avNNetModelC1V2, newdata = data2015)

avNNetC1V2_PR = postResample(pred = avNNetC1V2_Preds, obs = class2015)
avNNetC1V2_PR

# assess the performance of the predictions made by our 
# average neural nets model for the 2015 testing dataset via confusion matrix
avgNNc1v2_CFM <- confusionMatrix(data = avNNetC1V2_Preds, reference = class2015,
                                 positive = "Increase")
avgNNc1v2_CFM

avNNetC1V2_Prob <- predict(avNNetModelC1V2, data2015, type = "prob")

avNNetModelC1V2_ROC <- roc(response = class2015, 
                           predictor = avNNetC1V2_Prob$Increase,
                           levels = rev(levels(class2015)))

plot(avNNetModelC1V2_ROC, col = "red", lwd = 3, 
     main = "ROC curve for the Average Neural Net")

# calculate the area under the above ROC curve
auc_avgNN_C1V2 <- auc(avNNetModelC1V2_ROC)
cat('Area under the ROC curve for the Average Neural Net:', 
    round(auc_avgNN_C1V2, 3))













### Classification Forecasting Model #5: Support Vector Machine
library(kernlab)
set.seed(100)
svmGrid = expand.grid(.sigma = c(0, 0.01, 0.1), .C = 1:10)
set.seed(100)
svmC1V2.fit <- train(x = data2014, y = class2014,  method = "svmRadial",
                   preProc = c("center", "scale"), tuneGrid = svmGrid,
                   trControl = trainControl(method = "repeatedcv", 
                                            repeats = 5, classProbs =  TRUE))
svmC1V2.fit

# compare the expected classifications in 2015 to the observed classifications in 2015
svmC1V2_Preds = predict(svmC1V2.fit, newdata = data2015)

svmC1V2_PR = postResample(pred = svmC1_Preds, obs = class2015)
svmC1V2_PR

# assess the performance of our penalized classification model's predictions 
# via a confusion matrix
svmC1V2_CFM <- confusionMatrix(data = svmC1V2_Preds, reference = class2015,
                                positive = "Increase")
svmC1V2_CFM

svmC1V2_Prob <- predict(svmC1V2.fit, data2015, type = "prob")

svmC1V2_ROC <- roc(response = class2015, predictor = svmC1V2_Prob$Increase,
                 levels = rev(levels(class2015)))

plot(svmC1V2_ROC, col = "red", lwd = 3, main = "ROC curve for the SVM model")

# calculate the area under the above ROC curve
auc_svmC1V2 <- auc(svmC1V2_ROC)
cat('Area under the ROC curve for the Support Vector Machine model: ', 
    round(auc_svmC1V2, 3))











### Classification Forecasting Model #6: K-Nearest Neighbors 
set.seed(100)
knnC1V2 = train(x = data2014, y = class2014, method = "knn",
              preProc = c("center","scale"), tuneLength = 20)
knnC1V2

# compare the expected classifications in 2015 to the 
# observed classifications in 2015
knnC1V2_Preds = predict(knnC1V2, newdata = data2015)

knnC1V2_PR = postResample(pred = knnC1V2_Preds, obs = class2015)
knnC1V2_PR

knnC1V2_CFM <- confusionMatrix(data = knnC1V2_Preds, reference = class2015,
                               positive = "Increase")
knnC1V2_CFM

knnC1V2_Prob <- predict(knnC1V2, data2015, type = "prob")

knnC1V2_ROC <- roc(response = class2015, predictor = knnC1V2_Prob$Increase,
                 levels = rev(levels(class2015)))

plot(knnC1V2_ROC, col = "red", lwd = 3, 
     main = "ROC curve the KNN Model")

# calculate the area under the above ROC curve
auc_knnC1V2 <- auc(knnC1V2_ROC)
cat('Area under the ROC curve for the KNN model fit on the 2014 data: ', 
    round(auc_knnC1V2, 3))
