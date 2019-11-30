library(keras)


#install_tensorflow()

#install_keras()
# This will provide you with default CPU-based installations of Keras and TensorFlow.


#------------------------------Preparing the Data-------------------------------------------
# Here we load the dataset then create variables for our test and training data:

df <- read.csv("C:/Users/surya/Desktop/Breast Cancer Analysis/wisc_bc_data.csv")
df$diagnosis<-ifelse(df$diagnosis=='B', 0,1)


as.factor(df$diagnosis)
head(df)
str(df)

normalize <- function(x) {  
  ## ===============YOUR CODE HERE===============
  
  return ((x - min(x)) / (max(x) - min(x)))
  
  ## ============================================== (2 marks)
}

# Apply normalization to entire data frame, the normalized data is assigned to the
# 'BreastCancer_norm' variable

BreastCancer_norm <- as.data.frame(lapply(df, normalize)) 

set.seed(1) #can provide any number for seed
nall = nrow(BreastCancer_norm) #total number of rows in data
ntrain = floor(0.8 * nall) # number of rows for train,70%
ntest = floor(0.2* nall) # number of rows for test, 30%
index = seq(1:nall)
trainIndex = sample(index, ntrain) #train data set
testIndex = index[-trainIndex]

train = BreastCancer_norm[trainIndex,]
test = BreastCancer_norm[testIndex,]


# Size and format of data frame
X_train <- as.matrix(train)
y_train <- as.matrix(to_categorical(train$diagnosis))


X_test <- as.matrix(test)
y_test <- as.matrix(to_categorical(test$diagnosis))

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

#------------------------------Training and Evaluation-------------------------------------------
# Use the fit() function to train the model for 30 epochs using batches of 128 images:
history <- model %>% fit(
  X_train, y_train, 
  epochs = 100, batch_size = 5, 
  validation_split = 0.3
)

# The history object returned by fit() includes loss and accuracy metrics which we can plot:
plot(history)

# Evaluate the model's performance on the test data:
model %>% evaluate(X_test, y_test,verbose = 0)

# Calculating accuracy
predictions <- model %>% predict_classes(test)

# Confusion Matrix
test$diagnosis=as.integer(test$diagnosis)-1
table(factor(predictions, levels=min(test$diagnosis):max(test$diagnosis)),factor(test$diagnosis, levels=min(test$diagnosis):max(test$diagnosis)))

