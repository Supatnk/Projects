
#Both the libraries have SVM . This code use kernlab
#library(kernlab)
#library(e1071)

#####  Support Vector Machines Demo-------------------
## Example: Optical Character Recognition ----

## Step 1: Data investigation

## Step 2: Exploring and preparing the data ----
# read in data and examine structure
BreastCancer <- read.csv("C:/Users/surya/Desktop/Breast Cancer Analysis/wisc_bc_data.csv")
str(BreastCancer)

#BreastCancer is a dataframe 
#BreastCancer is outcome, rest fields are the features 

# divide into training and test data
BreastCancer_train <- BreastCancer[1:400, ]
BreastCancer_test  <- BreastCancer[401:569, ]

# better to partition the data for randomness 

## Step 3: Training a model on the data ----
# begin by training a simple linear SVM
#BreastCancer ~ .  => all other than BreastCancer field
library(kernlab)
BreastCancer_classifier <- ksvm(diagnosis ~ ., data = BreastCancer_train,
                          kernel = "vanilladot") # vanilla means linear kernel function

#ksvm is the specific syntax for kernlab 
# look at basic information about the model
BreastCancer_classifier

## Step 4: Evaluating model performance ----
# predictions on testing dataset
BreastCancer_predictions <- predict(BreastCancer_classifier, BreastCancer_test)

#head(BreastCancer_predictions)
head(BreastCancer_predictions,20)

table(BreastCancer_predictions, BreastCancer_test$diagnosis)

# look only at agreement vs. non-agreement
# construct a vector of TRUE/FALSE indicating correct/incorrect predictions
agreement <- BreastCancer_predictions == BreastCancer_test$diagnosis
table(agreement)
prop.table(table(agreement))

## Step 5: Improving model performance ----
# setting seed will give same result every time hence good for testing 
set.seed(100)
BreastCancer_classifier_rbf <- ksvm(diagnosis ~ ., data = BreastCancer_train, kernel = "rbfdot") #gaussian kernel 
BreastCancer_predictions_rbf <- predict(BreastCancer_classifier_rbf, BreastCancer_test)

agreement_rbf <- BreastCancer_predictions_rbf == BreastCancer_test$diagnosis
table(agreement_rbf)
prop.table(table(agreement_rbf))


#### Iris example
# just writing attach in command window . u dont have to write iris$fieldname. Onlye fieldname will be enough 
