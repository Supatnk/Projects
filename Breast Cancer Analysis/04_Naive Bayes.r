##### Chapter 4: Classification using Naive Bayes --------------------

## Example: Filtering B_group SMS messages ----
## Step 2: Exploring and preparing the data ---- 

# read the sms data into the sms data frame
Breast_Cancer_Raw <- read.csv("C:/Users/surya/Desktop/Breast Cancer Analysis/wisc_bc_data.csv", stringsAsFactors = FALSE)

# examine the structure of the sms data
str(Breast_Cancer_Raw)

# convert B_group/M_group to factor.
Breast_Cancer_Raw$diagnosis <- factor(Breast_Cancer_Raw$diagnosis)

# examine the type variable more carefully
str(Breast_Cancer_Raw$diagnosis)
table(Breast_Cancer_Raw$diagnosis)


# creating training and test datasets
Breast_Cancer_Train <- Breast_Cancer_Raw[1:400, ]
Breast_Cancer_Test  <- Breast_Cancer_Raw[401:569, ]

# also save the labels
Breast_Cancer_train_labels <- Breast_Cancer_Raw[1:400, ]$diagnosis
Breast_Cancer_test_labels  <- Breast_Cancer_Raw[401:569, ]$diagnosis

# check that the proportion of B_group is similar
prop.table(table(Breast_Cancer_train_labels))
prop.table(table(Breast_Cancer_test_labels))

# 
# subset the training data into B_group and M_group groups
B_group <- subset(Breast_Cancer_Raw, diagnosis == "B")
M_group  <- subset(Breast_Cancer_Raw, diagnosis == "M")


## Step 3: Training a model on the data ----
library(e1071)
sms_classifier <- naiveBayes(Breast_Cancer_Train, Breast_Cancer_train_labels)

## Step 4: Evaluating model performance ----
sms_test_pred <- predict(sms_classifier, Breast_Cancer_Test)

library(gmodels)
CrossTable(sms_test_pred, Breast_Cancer_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## Step 5: Improving model performance ----
Breast_Cancer_classifier2 <- naiveBayes(Breast_Cancer_Train, Breast_Cancer_train_labels, laplace = 1)
Breast_Cancer_pred2 <- predict(Breast_Cancer_classifier2, Breast_Cancer_Test)
CrossTable(Breast_Cancer_pred2, Breast_Cancer_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
