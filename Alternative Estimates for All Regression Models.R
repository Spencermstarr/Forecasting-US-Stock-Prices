##### Alternative estimates, i.e. alternately fitted predictive models
##### of the same types, just trained on the 2015, 2016, or 2017 data instead.




















### Regression Forecasting Model #4: Artificial Neural Network,
### but with positive weight decay values this time and also 
### with bagging!
library(nnet)

set.seed(100)
# exact same grid (for now)
NNetR1_Grid = expand.grid(.decay = c(0, 0.01, 0.1), .size = 1:10)

# fit/run a NN with only 1 hidden layers
set.seed(100)
NNet_R1_1layer = train(x = data2014, y = pr_var2014, method = "nnet", size = 1,
                       preProc = c("center", "scale"), linout = FALSE, trace = FALSE, 
                       MaxNWts = 10 * (ncol(data2014) + 1) + 10 + 1, maxit = 500, 
                       tuneGrid = NNetR1_Grid)
NNet_R1_1layer






### Regression Forecasting Model #4.5: Average (of several) Neural Networks,
### but with the bagging option & linout argument set to TRUE this time!
set.seed(100)
avgNNetModelR1_V2 = train(x = data2014, y = pr_var2014, method = "avNNet", 
                        bag = TRUE, preProc = c("center", "scale"), 
                        linout = TRUE, trace = FALSE, 
                        MaxNWts = 10 * (ncol(data2014) + 1) + 10 + 1, 
                        maxit = 500)
avgNNetModel_R1
summary(avgNNetModel_R1)
coef(avgNNetModel_R1)



set.seed(100)
avgNNetModelR2_V2 = train(x = data2015, y = pr_var2015, method = "avNNet", 
                          bag = TRUE, preProc = c("center", "scale"), 
                          linout = TRUE, trace = FALSE, 
                          MaxNWts = 10 * (ncol(data2015) + 1) + 10 + 1, 
                          maxit = 500)
avgNNetModel_R1
summary(avgNNetModel_R1)
coef(avgNNetModel_R1)












### Regression Forecasting Model #3: Multivariate Adaptive Regression Splines
library(earth)
library(Formula)
library(plotmo)
library(plotrix)
library(TeachingDemos)
# training a MARS on the 2015 stock data
marsGrid = expand.grid(.degree = 1:2, .nprune = 2:38)
set.seed(100)
## try using the train function from the caret package, and  
## setting the method argument equal to "earth".
marsModelR2 = train(x = data2015, y = pr_var2015, method = "earth", 
                    preProc = c("center", "scale"), tuneGrid = marsGrid)
marsModelR2

# compare the expected classifications in 2015 to the observed classifications in 2016
marsR2Pred = predict(marsModelR2, newdata = data2016)
length(marsR2Pred)
dim(marsR2Pred)
str(marsR2Pred)
marsR2Pred_2017 = predict(marsModelR2, newdata = data2017)  # same as above for 2017
marsR2Pred_2018 = predict(marsModelR2, newdata = data2018)  # same as above for 2018

marsR2_PR = postResample(pred = marsR2Pred, obs = pr_var2015)
marsR2_PR


## try using the earth function from the earth package instead
marsFits_2015 = earth(x = data2015, y = pr_var2015, penalty = 3)
marsFits_2015
summary(marsFits_2015)

marsPred_2015 = predict(marsFits_2015, newdata = data2016)
marsPred_2015

marsPR_2015 = postResample(pred = marsFits_2015, obs = pr_var2015)
marsPR_2015



# training a MARS on the 2016 stock data
set.seed(100)
marsModelR3 = train(x = data2016, y = pr_var2016, method = "earth", 
                    preProc = c("center", "scale"), tuneGrid = marsGrid)
marsModelR3

# compare the expected classifications in 2016 to the observed classifications in 2017
marsR3Pred = predict(marsModelR3, newdata = data2017)
length(marsR3Pred)
dim(marsR3Pred)
str(marsR3Pred)
marsR3Pred_2018 = predict(marsModelR3, newdata = data2018)  # same as above for 2018

marsR3_PR = postResample(pred = marsR3Pred, obs = pr_var2016)
marsR3_PR


## try using the earth function from the earth package instead
marsFits_2016 = earth(x = data2016, y = pr_var2016, penalty = 3)
marsFits_2016
summary(marsFits_2016)

marsPred_2016 = predict(marsFits_2016, newdata = data2017)
marsPred_2016

marsPR_2016 = postResample(pred = marsFits_2016, obs = pr_var2017)
marsPR_2016