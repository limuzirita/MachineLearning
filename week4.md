ML Course Project
================
Muzi Li
2/18/2020

The goal of this project is to predict the manner in which they did the
exercise.

``` r
# Create the data folder and define the directory for data
dataDir <- "./data"
if(!dir.exists(dataDir)){
    dir.create(dataDir)
}

# Define the data file
activity_train <- paste(dataDir, "activity_train.csv", sep="/")
activity_valid <- paste(dataDir, "activity_valid.csv", sep="/")

# Perform datafile download with the link
if(!file.exists(activity_train)){
    url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    download.file(url, destfile = activity_train)
}
if(!file.exists(activity_valid)){
    url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(url, destfile = activity_valid)
}
```

## Data Format

First, we need to format the data and process it into the useful form.

``` r
# Read data and format it
activity.Raw.train <- read.csv(activity_train, na.strings=c("NA","#DIV/0!", ""))
activity.Raw.valid <- read.csv(activity_valid, na.strings=c("NA","#DIV/0!", ""))

activity_train<-activity.Raw.train[,colSums(is.na(activity.Raw.train)) == 0]
activity_valid <-activity.Raw.valid[,colSums(is.na(activity.Raw.valid)) == 0]
```

## Data Split

Then, we need to control some columns that should not be used in the
modeling and split the data into 2 parts, train and valid in 70-30
manner using y.

``` r
# Check what are the control variables
# table(activity_train$new_window);table(activity_train$num_window)

activity_train <- activity_train[,-c(1:7)]
activity_valid <- activity_valid[,-c(1:7)]

activity <- activity_train
train_index <- createDataPartition(y=activity$classe, p=0.7, list=FALSE)
activity_train <- activity[train_index, ] 
activity_test <- activity[-train_index, ]
```

## Decision Tree

The first model used is decision tree. Fit the data with classification
and perform 10-fold cross validation to make the process more robustics
and reliable.

``` r
library(rpart)
dt <- rpart(classe ~., data = activity_train, method = 'class', xval = 10)
dt.pred = predict(dt, activity_test, type="class")
```

## Random Forest

The second model used is random forest. Fit the data with classification
and random forest itself is to produce many trees and make the result
with low variance.

``` r
library(randomForest)
#rf.cv <- rfcv(activity_train, activity_train$classe, cv.fold=10)
rf <- randomForest(classe ~. , data=activity_train, method="class")
rf.pred = predict(rf, activity_test, type="class")
```

## GBM

The third model used is GBM. Training data has been further split into
training and testing data lists. Fit the data with multi-classification
and perform 5-fold cross validation.

``` r
library(xgboost)
# Process data as xgboost friendly type
xgb_train_index <- createDataPartition(y=activity_train$classe, p=0.7, list=FALSE)
xgb_activity_train <- activity_train[xgb_train_index, ] 
xgb_activity_test <- activity_train[-xgb_train_index, ]
train.mat <-  xgb.DMatrix(as.matrix(xgb_activity_train %>% select(-classe)), label = xgb_activity_train$classe)
test.mat <- xgb.DMatrix(as.matrix(xgb_activity_test %>% select(-classe)), label = xgb_activity_test$classe)
watchlist <- list(train = train.mat, eval = test.mat)

# # prepare rounds for cv
 numberOfClasses <- length(unique(activity$classe))+1
 xgb_params <- list("objective" = "multi:softprob",
                    "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
# nround    <- 1000 # number of XGBoost rounds
# cv.nfold  <- 5
# 
# # Fit cv.nfold * cv.nround XGB models and save OOF predictions
# xgbcv <- xgb.cv(params = xgb_params,
#                    data = train.mat, 
#                    nrounds = nround,
#                    nfold = cv.nfold,
#                    verbose = FALSE,
#                    prediction = TRUE)
# which.min(xgbcv$evaluation_log[, test_mlogloss_mean])
xgb <- xgb.train (data = train.mat, nrounds = 994 , watchlist = watchlist, print_every_n = 100,
                  early_stop_round = 10, maximize = F , params=xgb_params)
```

    ## [1]  train-mlogloss:1.278949 eval-mlogloss:1.294761 
    ## [101]    train-mlogloss:0.001744 eval-mlogloss:0.016600 
    ## [201]    train-mlogloss:0.000811 eval-mlogloss:0.014182 
    ## [301]    train-mlogloss:0.000620 eval-mlogloss:0.013813 
    ## [401]    train-mlogloss:0.000534 eval-mlogloss:0.013702 
    ## [501]    train-mlogloss:0.000484 eval-mlogloss:0.013493 
    ## [601]    train-mlogloss:0.000453 eval-mlogloss:0.013456 
    ## [701]    train-mlogloss:0.000430 eval-mlogloss:0.013426 
    ## [801]    train-mlogloss:0.000412 eval-mlogloss:0.013328 
    ## [901]    train-mlogloss:0.000398 eval-mlogloss:0.013255 
    ## [994]    train-mlogloss:0.000387 eval-mlogloss:0.013190

``` r
valid.mat <- xgb.DMatrix(as.matrix(activity_test %>% select(-classe)), label = activity_test$classe)
xgb.pred <- predict(xgb, valid.mat, reshape=T)
xgb.pred = as.data.frame(xgb.pred[,-1])
colnames(xgb.pred) = levels(activity$classe)
xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])
```

### Below are the confusion Matrix for three models

#### For Decision Tree, the accuracy is 0.7524

``` r
confusionMatrix(dt.pred, activity_test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1544  178   21   61   11
    ##          B   47  602   66   81   95
    ##          C   46  170  847   89   89
    ##          D   23   73   79  646   81
    ##          E   14  116   13   87  806
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7553          
    ##                  95% CI : (0.7441, 0.7663)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6898          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9223   0.5285   0.8255   0.6701   0.7449
    ## Specificity            0.9356   0.9391   0.9189   0.9480   0.9521
    ## Pos Pred Value         0.8507   0.6756   0.6825   0.7162   0.7780
    ## Neg Pred Value         0.9681   0.8925   0.9615   0.9362   0.9431
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2624   0.1023   0.1439   0.1098   0.1370
    ## Detection Prevalence   0.3084   0.1514   0.2109   0.1533   0.1760
    ## Balanced Accuracy      0.9290   0.7338   0.8722   0.8091   0.8485

#### For Random Forest, the accuracy is 0.9939

``` r
confusionMatrix(rf.pred, activity_test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1672    1    0    0    0
    ##          B    2 1137    6    0    0
    ##          C    0    1 1016   16    3
    ##          D    0    0    4  948    4
    ##          E    0    0    0    0 1075
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9937          
    ##                  95% CI : (0.9913, 0.9956)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.992           
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9988   0.9982   0.9903   0.9834   0.9935
    ## Specificity            0.9998   0.9983   0.9959   0.9984   1.0000
    ## Pos Pred Value         0.9994   0.9930   0.9807   0.9916   1.0000
    ## Neg Pred Value         0.9995   0.9996   0.9979   0.9968   0.9985
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2841   0.1932   0.1726   0.1611   0.1827
    ## Detection Prevalence   0.2843   0.1946   0.1760   0.1624   0.1827
    ## Balanced Accuracy      0.9993   0.9983   0.9931   0.9909   0.9968

#### For XGB, the accuracy is 0.9932

``` r
confusionMatrix(as.factor(xgb.pred$prediction), activity_test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    2    0    0    0
    ##          B    0 1134    4    0    0
    ##          C    1    3 1019    8    0
    ##          D    0    0    3  956    5
    ##          E    0    0    0    0 1077
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9956          
    ##                  95% CI : (0.9935, 0.9971)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9944          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9994   0.9956   0.9932   0.9917   0.9954
    ## Specificity            0.9995   0.9992   0.9975   0.9984   1.0000
    ## Pos Pred Value         0.9988   0.9965   0.9884   0.9917   1.0000
    ## Neg Pred Value         0.9998   0.9989   0.9986   0.9984   0.9990
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2843   0.1927   0.1732   0.1624   0.1830
    ## Detection Prevalence   0.2846   0.1934   0.1752   0.1638   0.1830
    ## Balanced Accuracy      0.9995   0.9974   0.9954   0.9950   0.9977

#### Since the accuracy of random forest is 0.9939, which is the highest amomng three models, RF is selected.

``` r
predict_final = predict(rf, activity_valid, type="class")
```
