##### Cross Validation Demo: Classification using K Nearest Neighbors --------------------

## Example: Classifying Cancer Samples ----

# Step 1: Import the CSV format data "wisc_bc_data.csv"


wbcd <- read.csv("C:/Users/surya/Desktop/Breast Cancer Analysis/wisc_bc_data.csv", stringsAsFactors = TRUE)


############ Exploratory Data Analysis ############

# Step 2: Explore the data, e.g. examine the structure of the wbcd data frame
str(wbcd)

library(ggplot2)
ggplot(wbcd, aes(x = diagnosis, fill = diagnosis)) +
  geom_bar()

# check for NA/-Inf/Inf values  
apply(wbcd, 2, function(x) any(is.na(x) | is.infinite(x)))

# graphs 

# correlation Matrix
library(corrplot)
mydata.cor = cor(DeveSubset[,1:30], method = c("spearman"))
corrplot(mydata.cor, order = "AOE")

# HeatMap
palette = colorRampPalette(c("green", "white", "red")) (20)
heatmap(x = mydata.cor, col = palette, symm = TRUE)

############Exploratory Data Analysis end############

# drop the id featuren which is useless in this case
wbcd <- wbcd[-1]

# table of diagnosis (We ignore data balancing process)
table(wbcd$diagnosis)

# recode diagnosis as a factor
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"),
                         labels = c("0", "1"))

# table or proportions with more informative labels
round(prop.table(table(wbcd$diagnosis)) * 100, digits = 1)

# summarize three numeric features
summary(wbcd[c("radius_mean", "area_mean", "smoothness_mean")])

# create normalization function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}


# normalize the wbcd data
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
wbcd_n$diagnosis <- wbcd$diagnosis

# confirm that normalization worked
summary(wbcd_n$area_mean)

# Step 3: Split the data into developing subset (including training and validation subset) and testing subset
# pick up Malignant subset and Benign subset
MalignantSubset <- wbcd_n[(wbcd_n$diagnosis == 1), ]
BenignSubset <- wbcd_n[(wbcd_n$diagnosis == 0), ]

# create developing subset (80%) and testing subset (20%) in terms of Malignant and Benign respectively
smp1 <- floor(nrow(MalignantSubset)*4/5)
smp0 <- floor(nrow(BenignSubset)*4/5)

set.seed(1)
# Randomly generate smp1 indices for Malignant subset
idx1 <- sample(seq_len(nrow(MalignantSubset)), size = smp1) 
Malignant_deve <- MalignantSubset[idx1, ]
Malignant_test <- MalignantSubset[-idx1, ]

set.seed(0)
# Randomly generate smp0 indices for Benign subset
idx0 <- sample(seq_len(nrow(BenignSubset)), size = smp0) 
Benign_deve <- BenignSubset[idx0, ]
Benign_test <- BenignSubset[-idx0, ]

# Combine Malignant_test and Benign_test by rows as TestingSubset, which will be used to test the validated classifier
TestingSubset <- rbind(Malignant_test, Benign_test)
DeveSubset <- rbind(Malignant_deve, Benign_deve)
str(TestingSubset)
str(DeveSubset)

# Set 9/10 of the developing subset to be training subset
# the remaining 1/10 of the developing subset to be validation subset
fold <- 10 # 10-fold cross validation
smp1 <- floor(nrow(Malignant_deve)*9/10) 
smp0 <- floor(nrow(Benign_deve)*9/10) 
# Declare the 'train_ind' and 'corr' variables before using them
MalignantTrain_ind <- matrix(NA, nrow = fold, ncol = smp1) 
BenignTrain_ind <- matrix(NA, nrow = fold, ncol = smp0) 

#-Randomly sampling training indices ('train_ind[i,]') for each fold from developing subset
for (i in 1:fold){ 
  set.seed(i) # to guarantee repeatable results
  MalignantTrain_ind[i,] <- sample(seq_len(nrow(Malignant_deve)), size = smp1)
  
  set.seed(i*100)
  BenignTrain_ind[i,] <- sample(seq_len(nrow(Benign_deve)), size = smp0)
}

################################ (1) Knn Start ################################

## Step 4: Train a model/classifier on the training subset and valide the classifier on validation seubset by 'fold'-fold cross validation----

# load the "class" library where the knn function is going to be used
library(class) 

N <- 20 # suppose we are trying 1~20 nearest neighbours
BA <- matrix(NA, nrow = 1, ncol = fold) # BA: Balanced accuracy for 'fold' folds in terms of a specific k
BA_K <- matrix(NA, nrow = N, ncol = 1)  # BA_K: Store average BA of 'fold' folds for each k

library(descr)

for (k in 1:N)
{
  for (i in 1:fold)
  {
    TrainSubset <- rbind(Malignant_deve[MalignantTrain_ind[i,],], Benign_deve[BenignTrain_ind[i,],])
    ValidSubset <- rbind(Malignant_deve[-MalignantTrain_ind[i,],], Benign_deve[-BenignTrain_ind[i,],])
    # Randomly reorder the elements of the training subset
    set.seed(i*k)
    TrainSubset <- TrainSubset[sample(nrow(TrainSubset)),]
    # Randomly reorder the elements of the validation subset
    set.seed(i*k+100)
    ValidSubset <- ValidSubset[sample(nrow(ValidSubset)),]
    # Train the knn classifier with different k
    valid_pred <- knn(train = TrainSubset[,1:30], test = ValidSubset[,1:30], cl = TrainSubset[,31], k)
    # Get the confusion matrix table
    CT <- CrossTable(x = ValidSubset[,31], y = valid_pred)
    BA[i] <- (CT$t[1,1]/(CT$t[1,1]+CT$t[1,2]) + CT$t[2,2]/(CT$t[2, 1]+CT$t[2,2]))/2
  }
  BA_K[k] <- mean(BA)
}

# Find the max balanced accuracy with a specific k
max_k <- which.max(BA_K) 
print(max_k)
# Step 5: Test the classifer with trained k 
DeveSubset <- rbind(Malignant_deve, Benign_deve)
# Randomly reorder the elements of the developing subset
set.seed(12345)
DeveSubset <- DeveSubset[sample(nrow(DeveSubset)),]


deve_pred <- knn(train = DeveSubset[,1:30], test = TestingSubset[,1:30], cl = DeveSubset[,31], max_k)

##METRICS##

CT <- CrossTable(x = TestingSubset[,31], y = deve_pred)
BA_Test <- (CT$t[1,1]/(CT$t[1,1]+CT$t[1,2]) + CT$t[2,2]/(CT$t[2, 1]+CT$t[2,2]))/2

# look only at agreement vs. non-agreement
# construct a vector of TRUE/FALSE indicating correct/incorrect predictions
agreement_kNN <- deve_pred == TestingSubset$diagnosis
table(agreement_kNN)
prop.table(table(agreement_kNN))

################################ (2) SVM Start ################################


# begin by training a simple linear SVM
#BreastCancer ~ .  => all other than BreastCancer field
library(kernlab)
BreastCancer_classifier_svm <- ksvm(diagnosis ~ ., data = DeveSubset,
                                kernel = "vanilladot") # vanilla means linear kernel function

#ksvm is the specific syntax for kernlab 
# look at basic information about the model
print(BreastCancer_classifier_svm)

## Step 4: Evaluating model performance ----
# predictions on testing dataset
BreastCancer_predictions_SVM <- predict(BreastCancer_classifier_svm, TestingSubset)


#head(BreastCancer_predictions)
head(BreastCancer_predictions_SVM,20)

##METRICS##

table(BreastCancer_predictions_SVM, TestingSubset$diagnosis)
CT_SVM <- CrossTable(x = BreastCancer_predictions_SVM, y = TestingSubset$diagnosis)
BA_Test_SVM <- (CT_SVM$t[1,1]/(CT_SVM$t[1,1]+CT_SVM$t[1,2]) + CT_SVM$t[2,2]/(CT_SVM$t[2, 1]+CT_SVM$t[2,2]))/2

# look only at agreement vs. non-agreement
# construct a vector of TRUE/FALSE indicating correct/incorrect predictions
agreement <- BreastCancer_predictions_SVM == TestingSubset$diagnosis
table(agreement)
prop.table(table(agreement))


## Step 5: Improving model performance ----
# setting seed will give same result every time hence good for testing 
set.seed(100)
BreastCancer_classifier_rbf <- ksvm(diagnosis ~ ., data = DeveSubset, kernel = "rbfdot") #gaussian kernel 
BreastCancer_predictions_rbf <- predict(BreastCancer_classifier_rbf, TestingSubset)

table(BreastCancer_predictions_rbf, TestingSubset$diagnosis)
CT_SVM_RBF <- CrossTable(x = BreastCancer_predictions_rbf, y = TestingSubset$diagnosis)
BA_Test_SVM_RBF <- (CT_SVM_RBF$t[1,1]/(CT_SVM_RBF$t[1,1]+CT_SVM_RBF$t[1,2]) + CT_SVM_RBF$t[2,2]/(CT_SVM_RBF$t[2, 1]+CT_SVM_RBF$t[2,2]))/2

# look only at agreement vs. non-agreement
# construct a vector of TRUE/FALSE indicating correct/incorrect predictions
agreement_rbf <- BreastCancer_predictions_rbf == TestingSubset$diagnosis
table(agreement_rbf)
prop.table(table(agreement_rbf))

################################ SVM END ################################

########################### (3) Naive Bayes Start ###########################
## Training model using Naive Bayes ----
library(e1071)
sms_classifier <- naiveBayes(DeveSubset[,1:30], DeveSubset[,31])

## Evaluating model performance ----
Breast_Cancer_pred <- predict(sms_classifier, TestingSubset[,1:30])

library(gmodels)
CrossTable(Breast_Cancer_pred, TestingSubset[,31],
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

##METRICS##
table(Breast_Cancer_pred, TestingSubset$diagnosis)
CT_Naive_Bayes <- CrossTable(x = Breast_Cancer_pred, y = TestingSubset$diagnosis)
BA_Test_NB <- (CT_Naive_Bayes$t[1,1]/(CT_Naive_Bayes$t[1,1]+CT_Naive_Bayes$t[1,2]) + CT_Naive_Bayes$t[2,2]/(CT_Naive_Bayes$t[2, 1]+CT_Naive_Bayes$t[2,2]))/2

# look only at agreement vs. non-agreement
# construct a vector of TRUE/FALSE indicating correct/incorrect predictions
agreement <- sms_test_pred == TestingSubset$diagnosis
table(agreement)
prop.table(table(agreement))

## Improving model performance ----
Breast_Cancer_classifier2 <- naiveBayes(DeveSubset[,1:30], DeveSubset[,31], laplace = 3)
Breast_Cancer_pred2 <- predict(Breast_Cancer_classifier2, TestingSubset[,1:30])
CrossTable(Breast_Cancer_pred2, TestingSubset[,31],
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

##METRICS##
table(Breast_Cancer_pred2, TestingSubset$diagnosis)
CT_Naive_Bayes2 <- CrossTable(x = Breast_Cancer_pred2, y = TestingSubset$diagnosis)
BA_Test_NB2 <- (CT_Naive_Bayes2$t[1,1]/(CT_Naive_Bayes2$t[1,1]+CT_Naive_Bayes2$t[1,2]) + CT_Naive_Bayes2$t[2,2]/(CT_Naive_Bayes2$t[2, 1]+CT_Naive_Bayes2$t[2,2]))/2

# look only at agreement vs. non-agreement
# construct a vector of TRUE/FALSE indicating correct/incorrect predictions
agreement <- Breast_Cancer_pred2 == TestingSubset$diagnosis
table(agreement)
prop.table(table(agreement))

########################### Naive Bayes End ###########################


###################### (4) Neural Networks Start #####################

library(keras)
# Size and format of data frame
X_train <- as.matrix(DeveSubset)
y_train <- as.matrix(to_categorical(DeveSubset$diagnosis))


X_test <- as.matrix(TestingSubset)
y_test <- as.matrix(to_categorical(TestingSubset$diagnosis))

#------------------------------Defining the Model-------------------------------------------
# The core data structure of Keras is a model, a way to organize layers. 
# The simplest type of model is the sequential model, a linear stack of layers.
# We begin by creating a sequential model and then adding layers using the pipe (%>%) operator:
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = "relu", input_shape = ncol(X_train)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 75, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2, activation = "sigmoid")

#Use the summary() function to print the details of the model:
summary(model)

# Next, compile the model with appropriate loss function, optimizer, and metrics:
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# Use the fit() function to train the model for 20 epochs using batches of 5 :
history <- model %>% fit(
  X_train, y_train, 
  epochs = 20, batch_size = 20, 
  validation_split = 0.3
)

# The history object returned by fit() includes loss and accuracy metrics which we can plot:
plot(history)

# Evaluate the model's performance on the test data:
model %>% evaluate(X_test, y_test,verbose = 0)

# Calculating accuracy
predictions <- model %>% predict_classes(X_test)

# Confusion Matrix
#test$diagnosis=as.integer(test$diagnosis)-1
#table(factor(predictions, levels=min(test$diagnosis):max(test$diagnosis)),factor(test$diagnosis, levels=min(test$diagnosis):max(test$diagnosis)))

##METRICS##
table(predictions, TestingSubset$diagnosis)
CT_Neural_Network <- CrossTable(x = predictions, y = TestingSubset$diagnosis)
BA_Neural_Network <- (CT_Neural_Network$t[1,1]/(CT_Neural_Network$t[1,1]+CT_Neural_Network$t[1,2]) + CT_Neural_Network$t[2,2]/(CT_Neural_Network$t[2, 1]+CT_Neural_Network$t[2,2]))/2

# look only at agreement vs. non-agreement
# construct a vector of TRUE/FALSE indicating correct/incorrect predictions
agreement <- predictions == TestingSubset$diagnosis
table(agreement)
prop.table(table(agreement))


######################## Neural Networks End ########################

###################### (5) Decision Trees Start #####################

library(rpart)
library(rpart.plot)
library(caTools)


model_dtree<- rpart(diagnosis ~ ., data=DeveSubset)       #Implementing Decision Tree
preds_dtree <- predict(model_dtree,newdata=TestingSubset, type = "class")

rpart.plot(model_dtree, extra = 100)

plot(preds_dtree, main="Decision tree created using rpart")
(conf_matrix_dtree <- table(preds_dtree, TestingSubset$diagnosis))

##METRICS##
table(preds_dtree, TestingSubset$diagnosis)
CT_Decision_Tree <- CrossTable(x = preds_dtree, y = TestingSubset$diagnosis)
BA_Decision_Tree <- (CT_Decision_Tree$t[1,1]/(CT_Decision_Tree$t[1,1]+CT_Decision_Tree$t[1,2]) + CT_Decision_Tree$t[2,2]/(CT_Decision_Tree$t[2, 1]+CT_Decision_Tree$t[2,2]))/2

# look only at agreement vs. non-agreement
# construct a vector of TRUE/FALSE indicating correct/incorrect predictions
agreement <- preds_dtree == TestingSubset$diagnosis
table(agreement)
prop.table(table(agreement))

######################## Decision Trees End ########################


###################### (6) Logistic  Regression Start #####################

library(magrittr) # enables to use pipe operators
library(caret) # for pre-processing/cross-validation and ml algrythmes
library(ROCR) # for ROC curve and AUROC
library(ggplot2)

cancerFitAll <- glm(diagnosis ~., 
                    family = binomial(link = "logit"), 
                    data = DeveSubset)
summary(cancerFitAll)

varImp(cancerFitAll)

fitControl <- trainControl(method = "repeatedcv",
                           #number of folds is 10 by default
                           repeats = 3, 
                           savePredictions = T)

glmCancerFit <- train(diagnosis ~., 
                      data = DeveSubset,
                      method = "glm",
                      family = "binomial",
                      trControl = fitControl)

glmFitAcc <- train(diagnosis ~.,  
                   data = TestingSubset,
                   method = "glm",
                   metric = "Accuracy",
                   trControl = fitControl) %>% print


######################## Regression  End ########################

