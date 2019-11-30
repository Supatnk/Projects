library(rpart)
library(rpart.plot)
library(caTools)

BreastCancer <- read.csv("C:/Users/surya/Desktop/Breast Cancer Analysis/wisc_bc_data.csv")
df$BreastCancer<-ifelse(df$BreastCancer=='B', 0,1)


as.factor(BreastCancer$diagnosis)


set.seed(5)    
split=sample.split(BreastCancer, SplitRatio = 0.7)  # Splitting data into training and test dataset
training_set=subset(BreastCancer,split==TRUE)       # Training dataset
test_set=subset(BreastCancer,split==FALSE)          # Test dataset
dim(training_set)                                   # Dimenstions of training dataset

set.seed(42)
# model_dtree <- rpart(diagnosis ~ radius_mean +
#                        texture_mean +
#                        perimeter_mean +
#                        area_mean +
#                        smoothness_mean +
#                        compactness_mean +
#                        concavity_mean +
#                        points_mean +
#                        symmetry_mean +
#                        dimension_mean,
#             data = training_set,
#             method = "class")


model_dtree<- rpart(diagnosis ~ ., data=training_set)       #Implementing Decision Tree
preds_dtree <- predict(model_dtree,newdata=test_set, type = "class")

rpart.plot(model_dtree, extra = 100)

plot(preds_dtree, main="Decision tree created using rpart")
(conf_matrix_dtree <- table(preds_dtree, test_set$diagnosis))