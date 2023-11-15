##### Group 4's Collective Project Code
### Part 1: setting up the environment and loading the necessary packages
library(caret)
library(earth)
library(vip)
library(plyr)
library(readr)
library(pROC)
library(nnet)
library(parallel)
library(doParallel)
library(ranger)
library(caretEnsemble)

#allow multithreaded operating
threads <- detectCores()
cluster <- makePSOCKcluster(threads)
registerDoParallel(cluster)


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
company2014 <- data2014$Stock.Ticker
class2014 <- data2014$Class
sector2014 <- data2014$Sector

company2015 <- data2015$Stock.Ticker
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
data2014 <- subset(data2014, select = -c(Stock.Ticker, Class, Sector))
data2015 <- subset(data2015, select = -c(Stock.Ticker, Class, Sector))



# find and remove highly correlated predictors
correlations <- cor(data2014)

# find them
highCorr <- findCorrelation(correlations, cutoff = .8)
length(highCorr)
# remove them
data2014 <- data2014[, -highCorr]
data2015 <- data2015[, -highCorr]


# remove stock price variance column to prevent perfect multicollinearity
data2014 <- subset(data2014, select = -c(X2015.PRICE.VAR....))
data2015 <- subset(data2015, select = -c(X2016.PRICE.VAR....))


# define our model controls
ctrl <- trainControl(method = "LGOCV", summaryFunction = twoClassSummary,
                     classProbs = TRUE)









### Part 3: creating several financial forecasting models which 
###         generate objectively comparable predictions. 
###         Important note: when our comments say that some predict() is comparing
###         the expected classification in 2015 vs the observed classification in 
###         2015, that just means the degree to which stocks our model predicted would go up
###         (in 2015 based on how it was trained & cross-validated on the 2014 data)
###         actually went up. And likewise, the degree to which stocks our model 
###         predicted would go down in 2015 actually did go down.

## Random Forest version 1:
# Can try/explore m options of (1, 2, 3, ..., sqrt(111) = 10.5), 
# grows 500 trees.
set.seed(100)  # use the same seed for every model
# Define the Tuning Grid
rf1Grid <- expand.grid(.mtry = c(1:sqrt(ncol(data2014)), by = 2))  # sqrt of total number of variables is a common choice

# Train the Random Forest Model using the caret package
system.time( ftRF1 <- train(x = data2014, y = class2014, method = "rf", 
                            tuneGrid = rf1Grid, metric = "ROC", 
                            trControl = ctrl) )
# model summary
ftRF1
# Check the final model parameters
ftRF1$finalModel

# use the model fitted on the 2014 data to predict the 2015 data
RF1predict <- predict(ftRF1, newdata = data2015)

## Performance Assessment via Confusion Matrix, ROC curve, and AUC
# create the Confusion Matrix
RF1_CFM <- confusionMatrix(data = RF1predict, reference = class2015, 
                           positive = "Increase")

# Construct the ROC Curve and then calculate the Area Under the Curve (AUC)
# for the initial simple RF (with 500 trees).
RF1prob <- predict(ftRF1, newdata = data2015, type = "prob")
ROC_RF1 <- roc(response = class2015, predictor = RF1prob$Increase,
               levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF1, col = "red", lwd = 3, 
     main = "ROC curve for the initial Random Forest Model")

# calculate the Area Under the ROC
RF1_auc <- auc(ROC_RF1)
cat('Area under the ROC curve for the initial Random Forest Model:', 
    round(RF1_auc, 4), '\n')

# print out the Confusion Matrix
RF1_CFM

# variable importance evaluation for the RF with 500 trees
impRF1 <- varImp(ftRF1)
plot(impRF1, top = 10, main = 'Variable Importance Plot for the initial Random Forest Model')









## Random Forest version 2: 
# You can also manually specify that you want it to run 1,000 trees,
# not the 500 which it runs by default. That is done below.
# Set the seed for reproducibility
set.seed(100)
tuneGridRF1 <- expand.grid(.mtry = seq(1, sqrt(ncol(data2014)), by = 2),
                           .splitrule = "gini", .min.node.size = 1)

# Train the Random Forest model with 1000 trees
system.time( ftRF_1000 <- train(x = data2014, y = class2014, method = "ranger",
                                metric = "ROC", tuneGrid = tuneGridRF1,
                                num.trees = 1000,  # Set the number of trees to 1000
                                trControl = ctrl, preProcess = c("center", "scale"),
                                importance = "impurity") )

# Model summary
ftRF_1000
# Check the final model parameters
ftRF_1000$finalModel

# use the model fitted on the 2014 data to predict the 2015 data
RF_1000_predict <- predict(ftRF_1000, newdata = data2015)

## Performance Assessment via Confusion Matrix, ROC curve, and AUC
# confusion matrix for the RF with 1,000 trees
RF_1000_CFM <- confusionMatrix(data = RF_1000_predict, 
                               reference = class2015, 
                               positive = "Increase")


# Now construct the ROC Curve and calculate the AUC for the RF with 1,0000 trees
RF_1000_prob <- predict(ftRF_1000, newdata = data2015, type = "prob")
ROC_RF_1000 <- roc(response = class2015, predictor = RF_1000_prob$Increase,
                   levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF_1000, col = "red", lwd = 3, 
     main = "ROC curve for the 2nd Random Forest")

# calculate the Area Under the ROC
RF2_auc <- auc(ROC_RF_1000)
# Modified cat() function to include a newline character at the end
cat('Area under the ROC curve the 2nd version of Random Forest:', 
    round(RF2_auc, 4), '\n')

# print out the Confusion Matrix
RF_1000_CFM

# variable importance evaluation for the RF with 1000 trees
impRF2 <- varImp(ftRF_1000)
plot(impRF2, top = 10, 
     main = 'Variable Importance Plot for the 2nd Random Forest Model')

# variable importance evaluation for the RF with 1000 trees
#impRF_1000 <- varImp(ftRF_1000)
#plot(impRF_1000, top = 10, 
#     main = 'Variable Importance Plot for our Random Forest Model with 1,000 trees')











## Random Forest version 3: 
# New tuning strategies introduced are an extra possible splitrule to explore,
# namely, "extratrees", larger maximum potential m options to explore, and
# more minimum node size options to explore.
# Set the seed for reproducibility
set.seed(100)
# Create a tuning grid
tuneGridRF2 <- expand.grid(.mtry = c(1, 5, 10, 15, 20),
                           .splitrule = c("gini", "extratrees"),
                           .min.node.size = c(1, 5, 10))

# Train the Random Forest model
system.time( ftRF_tuned2 <- train(x = data2014, y = class2014, method = "ranger",
                     metric = "ROC", tuneGrid = tuneGridRF2,
                     num.trees = 500,  # You can adjust this based on your findings
                     trControl = ctrl, preProcess = c("center", "scale"),
                     importance = "impurity") )

# Model summary
ftRF_tuned2
# Check the final model parameters
ftRF_tuned2$finalModel

# use the model fitted on the 2014 data to predict the 2015 data
RF_tuned2_predict <- predict(ftRF_tuned2, newdata = data2015)

## Performance Assessment for the 3rd version of RF via 
## Confusion Matrix, ROC curve, and AUC.
# create the confusion matrix
RF_tuned2_CFM <- confusionMatrix(data = RF_tuned2_predict, 
                                 reference = class2015, positive = "Increase")

# Construct the ROC Curve and then calculate the Area Under the Curve (AUC)
# for the RF with 500 trees.
RF_tuned2_prob <- predict(ftRF_tuned2, newdata = data2015, type = "prob")
ROC_RF_tuned2 <- roc(response = class2015, predictor = RF_tuned2_prob$Increase,
                     levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF_tuned2, col = "red", lwd = 3, 
     main = "ROC curve for the 3rd Random Forest")

# calculate the Area Under the ROC
RF3_auc <- auc(ROC_RF_tuned2)
# Modified cat() function to include a newline character at the end
cat('Area under the ROC curve for variation #3 on the Random Forest model:', 
    round(RF3_auc, 4), '\n')

# print out the Confusion Matrix
RF_tuned2_CFM

# variable importance evaluation for the 3rd RF
impRF3 <- varImp(ftRF_tuned2)
plot(impRF3, top = 10, 
     main = 'Variable Importance Plot for the 3rd Random Forest Model')













## Random Forest version 4: 
# Try again using the same tuning grid as RF version 3, but 
# use 1,000 trees instead of 500 this time.
# Train the Random Forest model
ftRF_tuned3 <- train(x = data2014, y = class2014, method = "ranger",
                     metric = "ROC", tuneGrid = tuneGridRF2,
                     num.trees = 1000,  # You can adjust this based on your findings
                     trControl = ctrl, preProcess = c("center", "scale"),
                     importance = "impurity")

# Model summary
ftRF_tuned3
# Check the final model parameters
ftRF_tuned3$fintalModel


# use the model fitted on the 2014 data to predict the 2015 data
RF_tuned3_predict <- predict(ftRF_tuned3, newdata = data2015)

## Performance Assessment for the 4th version of RF via 
## Confusion Matrix, ROC curve, and AUC
# create the confusion matrix
RF_tuned3_CFM <- confusionMatrix(data = RF_tuned3_predict, reference = class2015, 
                                 positive = "Increase")

# Construct the ROC Curve and then calculate the Area Under the Curve (AUC)
# for the RF with 1,000 trees.
RF_tuned3_prob <- predict(ftRF_tuned3, newdata = data2015, type = "prob")
ROC_RF_tuned3 <- roc(response = class2015, predictor = RF_tuned3_prob$Increase,
                     levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF_tuned3, col = "red", lwd = 3, 
     main = "ROC curve for the 4th Random Forest")

# calculate the Area Under the ROC 
RF4_auc <- auc(ROC_RF_tuned3)
# Modified cat() function to include a newline character at the end
cat('Area under the ROC curve for variation #4 of the Random Forest model:', 
    round(RF4_auc, 4), '\n')

# print out the Confusion Matrix
RF_tuned3_CFM

# variable importance evaluation for the 4th RF
impRF4 <- varImp(ftRF_tuned3)
plot(impRF4, top = 10, 
     main = 'Variable Importance Plot for the 4th Random Forest Model')













## Random Forest version 5:
# New expansion of fine tuning is lowering the minimum
# node size from 1 down to 0.5.
# Set the seed for reproducibility
set.seed(100)
# Create a tuning grid
tuneGridRF3 <- expand.grid(.mtry = c(1, 5, 10, 15, 20),  # Same range for mtry
                           .splitrule = c("gini", "extratrees"),  # Same splitrules
                           .min.node.size = c(0.5, 1, 5, 10))  # Decrease minimum node size range to 0.5

# Train the Random Forest model
system.time( ftRF_tuned4 <- train(x = data2014, y = class2014, method = "ranger",
                                  metric = "ROC", tuneGrid = tuneGridRF3,
                                  num.trees = 1000,  # You can adjust this based on your findings
                                  trControl = ctrl, preProcess = c("center", "scale"),
                                  importance = "impurity") )

# Model summary
ftRF_tuned4
# Check the final model parameters
ftRF_tuned4$finalModel

# use the model fitted on the 2014 data to predict the 2015 data
RF_tuned4_predict <- predict(ftRF_tuned4, newdata = data2015)

## Performance Assessment for the 5th version of RF via 
## Confusion Matrix, ROC curve, and AUC
# create the confusion matrix
RF_tuned4_CFM <- confusionMatrix(data = RF_tuned4_predict, reference = class2015, 
                                 positive = "Increase")

# Construct the ROC Curve and then calculate the Area Under the Curve (AUC)
# for the RF with 500 trees.
RF_tuned4_prob <- predict(ftRF_tuned4, newdata = data2015, type = "prob")
ROC_RF_tuned4 <- roc(response = class2015, predictor = RF_tuned4_prob$Increase,
                     levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF_tuned4, col = "red", lwd = 3, 
     main = "ROC curve for the 5th Random Forest")

# calculate the Area Under the ROC 
RF5_auc <- auc(ROC_RF_tuned4)
# Modified cat() function to include a newline character at the end
cat('Area under the ROC curve for variation #5 on the Random Forest model:', 
    round(RF5_auc, 4), '\n')

# print out the Confusion Matrix
RF_tuned4_CFM

# variable importance evaluation for the 5th RF
impRF5 <- varImp(ftRF_tuned4)
plot(impRF5, top = 10, 
     main = 'Variable Importance Plot for the 5th Random Forest Model')












## Random Forest version 6: 
# Add a new splitrule option, namely, the hellinger splitrule.
# Set the seed for reproducibility
set.seed(100)
tuneGridRF4 <- expand.grid(
  .mtry = c(1, 5, 10, 15, 20),  # Same mtry range
  .splitrule = c("gini", "extratrees", "hellinger"),  # Added "hellinger"
  .min.node.size = c(0.5, 1, 5, 10)  # Same minimum node size range
)

# Train the Random Forest model
system.time( ftRF_tuned5 <- train(x = data2014, y = class2014, method = "ranger",
                                  metric = "ROC", tuneGrid = tuneGridRF4,
                                  num.trees = 1000,  # You can adjust this based on your findings
                                  trControl = ctrl, preProcess = c("center", "scale"),
                                  importance = "impurity"))

# Model summary
ftRF_tuned5
# Check the final model parameters
ftRF_tuned5$finalModel

# use the model fitted on the 2014 data to predict the 2015 data
RF_tuned5_predict <- predict(ftRF_tuned5, newdata = data2015)

## Performance Assessment for the 6th version of RF via 
## Confusion Matrix, ROC curve, and AUC
# create the confusion matrix
RF_tuned5_CFM <- confusionMatrix(data = RF_tuned5_predict, reference = class2015, 
                                 positive = "Increase")

# Construct the ROC Curve and then calculate the Area Under the Curve (AUC)
# for the RF with 1,000 trees.
RF_tuned5_prob <- predict(ftRF_tuned5, newdata = data2015, type = "prob")
ROC_RF_tuned5 <- roc(response = class2015, predictor = RF_tuned5_prob$Increase,
                     levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF_tuned5, col = "red", lwd = 3, 
     main = "ROC curve for the 6th Random Forest")

# calculate the Area Under the ROC 
RF6_auc <- auc(ROC_RF_tuned5)
# Modified cat() function to include a newline character at the end
cat('Area under the ROC curve for variation #6 of the Random Forest model:', 
    round(RF6_auc, 4), '\n')

# print out the Confusion Matrix
RF_tuned5_CFM

# variable importance evaluation for the 6th RF
impRF6 <- varImp(ftRF_tuned5)
plot(impRF6, top = 10, 
     main = 'Variable Importance Plot for the 6th Random Forest Model')















## Random Forest version 7: 
## This one is the same as the 6th variation, except with the 
## range of min.node.sizes available increased from 10 to 15.
# Set the seed for reproducibility
set.seed(100)
tuneGridRF5 <- expand.grid(
  .mtry = c(1, 5, 10, 15, 20),  # Same set of mtry options
  .splitrule = c("gini", "extratrees", "hellinger"),  # Added "hellinger"
  .min.node.size = c(0.5, 1, 5, 10, 15)  # Increase the max minimum node size option from 10 to 15
)

# Train the Random Forest model
system.time( ftRF_tuned6 <- train(x = data2014, y = class2014, method = "ranger",
                                  metric = "ROC", tuneGrid = tuneGridRF5,
                                  num.trees = 1000,  # You can adjust this based on your findings
                                  trControl = ctrl, preProcess = c("center", "scale"),
                                  importance = "impurity") )

# Model summary
ftRF_tuned6
# Check the final model parameters
ftRF_tuned6$finalModel

# use the model fitted on the 2014 data to predict the 2015 data
RF_tuned6_predict <- predict(ftRF_tuned6, newdata = data2015)

## Performance Assessment for the 7th version of RF via 
## Confusion Matrix, ROC curve, and AUC
# create the confusion matrix
RF_tuned6_CFM <- confusionMatrix(data = RF_tuned6_predict, reference = class2015, 
                                 positive = "Increase")

# Construct the ROC Curve and then calculate the Area Under the Curve (AUC)
RF_tuned6_prob <- predict(ftRF_tuned6, newdata = data2015, type = "prob")
ROC_RF_tuned6 <- roc(response = class2015, predictor = RF_tuned6_prob$Increase,
                     levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF_tuned6, col = "red", lwd = 3, 
     main = "ROC curve for the 7th Random Forest")

# calculate the Area Under the ROC
RF7_auc <- auc(ROC_RF_tuned6)
# Modified cat() function to include a newline character at the end
cat('Area under the ROC curve for the 7th Random Forest variation:', 
    round(RF7_auc, 4), '\n')

# print out the Confusion Matrix
RF_tuned6_CFM


# variable importance evaluation for the 7th RF
impRF7 <- varImp(ftRF_tuned6)
plot(impRF7, top = 10, 
     main = 'Variable Importance Plot for the 7th Random Forest Model')















## Random Forest version 8: 
## This one is the same as the 7th variation, except with the 
## maximum mtry available increased from 20 to 25.
# Set the seed for reproducibility
set.seed(100)
tuneGridRF6 <- expand.grid(
  .mtry = c(1, 5, 10, 15, 20, 25),  # Increase the maximum mtry option from 20 to 25
  .splitrule = c("gini", "extratrees", "hellinger"),  # Same three splitrule options
  .min.node.size = c(0.5, 1, 5, 10, 15)  # Same minimum node size range
)

# Train the Random Forest model
system.time( ftRF_tuned7 <- train(x = data2014, y = class2014, method = "ranger",
                                  metric = "ROC", tuneGrid = tuneGridRF6,
                                  num.trees = 1000,  # You can adjust this based on your findings
                                  trControl = ctrl, preProcess = c("center", "scale"),
                                  importance = "impurity") )

# Model summary
ftRF_tuned7
# Check the final model parameters
ftRF_tuned7$finalModel

# use the model fitted on the 2014 data to predict the 2015 data
RF_tuned7_predict <- predict(ftRF_tuned7, newdata = data2015)

## Performance Assessment for the 8th version of RF via 
## Confusion Matrix, ROC curve, and AUC
# create the confusion matrix
RF_tuned7_CFM <- confusionMatrix(data = RF_tuned7_predict, reference = class2015, 
                                 positive = "Increase")

# Construct the ROC Curve and then calculate the 
# Area Under the Curve (AUC) for the 8th RF variation.
RF_tuned7_prob <- predict(ftRF_tuned7, newdata = data2015, type = "prob")
ROC_RF_tuned7 <- roc(response = class2015, predictor = RF_tuned7_prob$Increase,
                     levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF_tuned7, col = "red", lwd = 3, 
     main = "ROC curve for 8th Random Forest")

# calculate the Area Under the ROC 
RF8_auc <- auc(ROC_RF_tuned7)
# Modified cat() function to include a newline character at the end
cat('Area under the ROC curve for the 8th Random Forest variation:', 
    round(RF8_auc, 4), '\n')

# print out the Confusion Matrix
RF_tuned7_CFM

# variable importance evaluation for the 8th RF
impRF8 <- varImp(ftRF_tuned7)
plot(impRF8, top = 10, 
     main = 'Variable Importance Plot for the 8th Random Forest Model')















## Random Forest version 9: 
## This one is an attempt to improve on version 6's performance by 
## bringing increasing the mtry range while also bringing back the 
## same minimum node size range.
# Set the seed for reproducibility
set.seed(100)
tuneGridRF7 <- expand.grid(
  .mtry = c(1, 5, 10, 15, 20, 25),  # Same maximum mtry options as the 8th RF
  .splitrule = c("gini", "extratrees", "hellinger"),  # Same three splitrule options as the 6th, 7th, & 8th RFs
  .min.node.size = c(0.5, 1, 5, 10)  # Same minimum node size range as the 6th RF 
)

# Train the Random Forest model
system.time( ftRF_tuned8 <- train(x = data2014, y = class2014, method = "ranger",
                                  metric = "ROC", tuneGrid = tuneGridRF7,
                                  num.trees = 1000,  # You can adjust this based on your findings
                                  trControl = ctrl, preProcess = c("center", "scale"),
                                  importance = "impurity") )

# Model summary
ftRF_tuned8
# Check the final model parameters
ftRF_tuned8$finalModel

# use the model fitted on the 2014 data to predict the 2015 data
RF_tuned8_predict <- predict(ftRF_tuned8, newdata = data2015)

## Performance Assessment for the 8th version of RF via 
## Confusion Matrix, ROC curve, and AUC
# create the confusion matrix
RF_tuned8_CFM <- confusionMatrix(data = RF_tuned8_predict, reference = class2015, 
                                 positive = "Increase")

# Construct the ROC Curve and then calculate the 
# Area Under the Curve (AUC) for the 8th RF variation.
RF_tuned8_prob <- predict(ftRF_tuned8, newdata = data2015, type = "prob")
ROC_RF_tuned8 <- roc(response = class2015, predictor = RF_tuned8_prob$Increase,
                     levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF_tuned8, col = "red", lwd = 3, 
     main = "ROC curve for 9th Random Forest")

# calculate the Area Under the ROC 
RF9_auc <- auc(ROC_RF_tuned8)
# Modified cat() function to include a newline character at the end
cat('Area under the ROC curve for the 9th Random Forest variation:', 
    round(RF9_auc, 4), '\n')

# print out the Confusion Matrix
RF_tuned8_CFM


# variable importance evaluation for the 9th RF
impRF9 <- varImp(ftRF_tuned8)
plot(impRF9, top = 10, 
     main = 'Variable Importance Plot for the 9th Random Forest Model')








































cat('Area under the ROC curve for the initial Random Forest:', 
    round(RF1_auc, 4), '\n')
# print out the Confusion Matrix
RF_CFM


cat('Area under the ROC curve for the 2nd version of Random Forest:', 
    round(RF2_auc, 4), '\n')
# print out the Confusion Matrix
RF_1000_CFM


cat('Area under the ROC curve for the 3rd version of Random Forest:', 
    round(RF3_auc, 4), '\n')
# print out the Confusion Matrix
RF_tuned2_CFM


cat('Area under the ROC curve for the 4th Random Forest variation:', 
    round(RF4_auc, 4), '\n')
# print out the Confusion Matrix
RF_tuned2_CFM


cat('Area under the ROC curve for the 5th Random Forest variation:', 
    round(RF5_auc, 4), '\n')
# print out the Confusion Matrix
RF_tuned5_CFM


cat('Area under the ROC curve for the 6th Random Forest variation:', 
    round(RF6_auc, 4), '\n')
# print out the Confusion Matrix
RF_tuned4_CFM


cat('Area under the ROC curve for the 7th Random Forest variation:', 
    round(RF7_auc, 4), '\n')
# print out the Confusion Matrix
RF_tuned5_CFM


cat('Area under the ROC curve for the 8th Random Forest variation:', 
    round(RF8_auc, 4), '\n')
# print out the Confusion Matrix
RF_tuned6_CFM


cat('Area under the ROC curve for the 9th Random Forest variation:', 
    round(RF7_auc, 4), '\n')
# print out the Confusion Matrix
RF_tuned7_CFM