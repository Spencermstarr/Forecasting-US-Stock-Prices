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


### assign all categorical variables to separate objects so
### that I can remove all non-numeric columns manually
# 2014
company2014 <- data2014$Stock.Ticker
class2014 <- data2014$Class
sector2014 <- data2014$Sector

# 2015
compary2015 <- data2015$Stock.Ticker
class2015 <- data2015$Class
sector2015 <- data2015$Sector


# use ifelse functions to distinguish between whether the price of 
# a stock increased or decreased during a given year using those words
class2014 <- ifelse(class2014 == 1, "Increase", "Decrease")
class2015 <- ifelse(class2015 == 1, "Increase", "Decrease")

# convert the integer values in the class column of both yearly stock market 
# datasets (which are both stored in their own dataframe) into factors stored in
# their own newly created objects separate from the dataframes they came from
class2014 <- as.factor(class2014)
class2015 <- as.factor(class2015)

# remove the Stock.Ticker, Class, & Sector columns from each annual dataframe 'manually'
data2014 <- subset(data2014, select = -c(Stock.Ticker, Class, Sector))
data2015 <- subset(data2015, select = -c(Stock.Ticker, Class, Sector))


### Interpolate the missing values in data2014 using their medians.
# find and count the position of all NAs (column-wise)
which(is.na(data2014))
# find all rows with at least 1 NA
which(rowSums(is.na(data2014)) != 0)
# count all NAs in data2014
sum(is.na(data2014))
# find/count the # of NAs in each column of data2014
head(colSums(is.na(data2014)))

## Apply median value interpolation to replace all missing values.
median_data2014 <- lapply(na.omit(data2014), median)
median_data2014 <- lapply(median_data2014, round)
for(i in 1:length(data2014)) {
  data2014[is.na(data2014[, i]), i] <- median(data2014[, i], na.rm = TRUE) }

# Now, count all NAs in data2014 again to verify that it worked!
sum(is.na(data2014))
# now remove the median_data_014 list since its purpose has been served
rm(median_data2014)


# median interpolation of all missing values in data2015
median_data2015 <- lapply(na.omit(data2015), median)
for(i in 1:length(data2015)) {
  data2015[is.na(data2015[, i]), i] <- median(data2015[, i], na.rm = TRUE) }

# now remove the median_data2015 list since its purpose has been served
rm(median_data2015)


dim(data2014)
dim(data2015)





# find and remove highly correlated predictors
correlations <- cor(data2014)

# find them
highCorr <- findCorrelation(correlations, cutoff = .8)
length(highCorr)

# remove them
data2014 <- data2014[, -highCorr]
data2015 <- data2015[, -highCorr]
dim(data2014)
dim(data2015)


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
###         2015, that just medians the degree to which stocks our model predicted would go up
###         (in 2015 based on how it was trained & cross-validated on the 2014 data)
###         actually went up. And likewise, the degree to which stocks our model 
###         predicted would go down in 2015 actually did go down.

## Ensemble Learning Model: Random Forest, version #1
set.seed(100)  # use the same seed for every model
# Define the Tuning Grid
tuneGridRF1 <- expand.grid(.mtry = seq(1, sqrt(ncol(data2014)), by = 2),
                           .splitrule = "gini", .min.node.size = 1)

# Train the Random Forest model with 500 trees
time_to_fit_RF1 <- system.time( ftRF1 <- train(x = data2014, y = class2014, method = "ranger",
                                               metric = "ROC", tuneGrid = tuneGridRF1,
                                               num.trees = 500, trControl = ctrl, 
                                               preProcess = c("center", "scale")) 
                                )
# model summary
ftRF1
# Check the final model parameters
ftRF1$finalModel

# use the model fitted on the 2014 data to predict the 2015 data
RF1preds <- predict(ftRF1, newdata = data2015)

# performance assessment for the RF with the default of 500 trees
RF1_CFM <- confusionMatrix(data = RF1preds, reference = class2015, 
                           positive = "Increase")

# Construct the ROC Curve and then calculate the Area Under the Curve (AUC)
# for the RF with 500 trees.
RF1prob <- predict(ftRF1, newdata = data2015, type = "prob")
ROC_RF1 <- roc(response = class2015, predictor = RF1prob$Increase,
               levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF1, col = "red", lwd = 3, 
     main = "ROC curve for the 1st Random Forest 
     ran again on better (median interpolated) data")

# calculate the Area Under the ROC
RF1_auc <- auc(ROC_RF1)
cat('Area under the ROC curve for our Random Forest model:', 
    round(RF1_auc, 4), '\n')

RF1_CFM

# variable importance evaluation for the RF with 500 trees
impRF1 <- varImp(ftRF1)
plot(impRF1, top = 10, 
     main = 'Variable Importance Plot for the 1st Random Forest Model')












## Random Forest version #2
# You can also manually specify that you want it to run 1,000 trees,
# not the 500 which it runs by default. That is done below.
# Define the tuning grid to use 1000 trees
# Set the seed for reproducibility
set.seed(100)
tuneGridRF2 <- expand.grid(.mtry = seq(1, sqrt(ncol(data2014)), by = 2),
                          .splitrule = "gini", .min.node.size = 1)

# Train the Random Forest model with 1000 trees
time_to_fit_RF2 <- system.time( ftRF_tuned3 <- train(x = data2014, y = class2014, method = "ranger",
                                                     metric = "ROC", tuneGrid = tuneGridRF2,
                                                     num.trees = 1000,  # Set the number of trees to 1000
                                                     trControl = ctrl, preProcess = c("center", "scale")) 
                                )
# Model summary
ftRF_tuned2
# Check the final model parameters
ftRF_tuned2$finalModel

# same for the RF with 1,000 trees now
RF2_predict <- predict(ftRF_tuned2, newdata = data2015)

# confusion matrix for the RF with 1,000 trees
RF2_CFM <- confusionMatrix(data = RF2_predict, 
                               reference = class2015, 
                               positive = "Increase")
RF2_CFM

# Now construct the ROC Curve and calculate the AUC for the RF with 1,0000 trees
RF2_prob <- predict(ftRF_tuned2, newdata = data2015, type = "prob")
ROC_RF2 <- roc(response = class2015, predictor = RF2_prob$Increase,
               levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF2, col = "red", lwd = 3, 
     main = "ROC curve for the Random Forest with 1,000 trees")

# calculate the Area Under the ROC
RF2_auc <- auc(ROC_RF2)
cat('Area under the ROC curve for our Random Forest with 1,000:', 
    round(RF2_auc, 4), '\n')

# variable importance evaluation for the RF with 1000 trees
impRF2 <- varImp(ftRF_tuned2)
plot(impRF2, top = 10, 
     main = 'Variable Importance Plot for the 2nd Random Forest Model')










## Random Forest version #3
# Set the seed for reproducibility
set.seed(100)
# Create a tuning grid
tuneGridRF3 <- expand.grid(.mtry = c(1, 5, 10, 15, 20),
                           .splitrule = c("gini", "extratrees"),
                           .min.node.size = c(1, 5, 10))

# Train the Random Forest model
time_to_fit_RF3 <- system.time( ftRF_tuned3 <- train(x = data2014, y = class2014, method = "ranger",
                                                     metric = "ROC", tuneGrid = tuneGridRF3,
                                                     num.trees = 500,  # You can adjust this based on your findings
                                                     trControl = ctrl, preProcess = c("center", "scale")) 
                                )
# Model summary
ftRF_tuned3
# Check the final model parameters
ftRF_tuned3$finalModel

# use the model fitted on the 2014 data to predict the 2015 data
RF_tuned3_predict <- predict(ftRF_tuned3, newdata = data2015)

# performance assessment for the RF with the default of 500 trees
RF_tuned3_CFM <- confusionMatrix(data = RF_tuned3_predict, reference = class2015, 
                                 positive = "Increase")

# Construct the ROC Curve and then calculate the Area Under the Curve (AUC)
# for the RF with 500 trees.
RF_tuned3_prob <- predict(ftRF_tuned3, newdata = data2015, type = "prob")
ROC_RF_tuned3 <- roc(response = class2015, predictor = RF_tuned3_prob$Increase,
                     levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF_tuned3, col = "red", lwd = 3, 
     main = "ROC curve for the 3rd differently tuned RF ran on the data  
              using Median imputation to handle missing values")

# calculate the Area Under the ROC
RF3_auc <- auc(ROC_RF_tuned3)
cat('Area under the ROC curve for variation #3 on the Random Forest model:', 
    round(RF3_auc, 4), '\n')

RF_tuned3_CFM












## Random Forest version #4
# Try again using the same tuning grid as RF version 3, but 
# use 1,000 trees instead of 500 this time.
# Train the Random Forest model
time_to_fit_RF4 <- system.time( ftRF_tuned4 <- train(x = data2014, y = class2014, method = "ranger",
                                                     metric = "ROC", tuneGrid = tuneGridRF3,
                                                     num.trees = 1000,  # You can adjust this based on your findings
                                                     trControl = ctrl, preProcess = c("center", "scale")) 
                                )
# Model summary
ftRF_tuned4
# Check the final model parameters
ftRF_tuned4$fintalModel

# use the model fitted on the 2014 data to predict the 2015 data
RF_tuned4_predict <- predict(ftRF_tuned4, newdata = data2015)

# performance assessment for the RF with the default of 500 trees
RF_tuned4_CFM <- confusionMatrix(data = RF_tuned4_predict, reference = class2015, 
                                 positive = "Increase")
RF_tuned4_CFM

# Construct the ROC Curve and then calculate the Area Under the Curve (AUC)
# for the RF with 1,000 trees.
RF_tuned4_prob <- predict(ftRF_tuned4, newdata = data2015, type = "prob")
ROC_RF_tuned4 <- roc(response = class2015, predictor = RF_tuned4_prob$Increase,
                     levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF_tuned4, col = "red", lwd = 3, 
     main = "ROC curve for the 4th differently tuned version of Random Forest")

# calculate the Area Under the ROC
RF4_auc <- auc(ROC_RF_tuned4)
cat('Area under the ROC curve for variation #4 of the Random Forest model: ', 
    round(RF4_auc, 4), '\n')











## RF variation #5
# Set the seed for reproducibility
set.seed(100)
tuneGridRF4 <- expand.grid(
  .mtry = c(1, 5, 10, 15, 20),  # Same upper range
  .splitrule = c("gini", "extratrees", "hellinger"),  # Added "hellinger"
  .min.node.size = c(0.5, 1, 5, 10)  # Lowered the minimum node size
)

# Train the Random Forest model
time_to_fit_RF5 <- system.time( ftRF_tuned5 <- train(x = data2014, y = class2014, method = "ranger",
                                                     metric = "ROC", tuneGrid = tuneGridRF4,
                                                     num.trees = 1000,  # You can adjust this based on your findings
                                                     trControl = ctrl, preProcess = c("center", "scale")) 
                                )
# Model summary
ftRF_tuned5
# Check the final model parameters
ftRF_tuned5$finalModel

# use the model fitted on the 2014 data to predict the 2015 data
RF_tuned5_predict <- predict(ftRF_tuned5, newdata = data2015)

# performance assessment for the 5th RF variant
RF_tuned5_CFM <- confusionMatrix(data = RF_tuned5_predict, reference = class2015, 
                                 positive = "Increase")

# Construct the ROC Curve and then calculate the Area Under the Curve (AUC)
RF_tuned5_prob <- predict(ftRF_tuned5, newdata = data2015, type = "prob")
ROC_RF_tuned5 <- roc(response = class2015, predictor = RF_tuned5_prob$Increase,
                     levels = rev(levels(class2015)))

# display the ROC curve just constructed
plot(ROC_RF_tuned5, col = "red", lwd = 3, 
     main = "ROC curve for the 5th RF ran on the data using 
              Median imputation to handle missing values")

# calculate the Area Under the ROC 
RF5_auc <- auc(ROC_RF_tuned5)

cat('Area under the ROC curve for variation #5 of the Random Forest model:', 
    round(RF5_auc, 4), '\n')

RF_tuned5_CFM









