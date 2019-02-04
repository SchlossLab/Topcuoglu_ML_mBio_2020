library(caret)

set.seed(1)
dat <- twoClassSim(100)

svm <- getModelInfo("svmLinear5")[[1]]

inTraining <- createDataPartition(data$dx, p = .80, list = FALSE)
training <- data[ inTraining,]
testing  <- data[-inTraining,]
# Scale all features between 0-1
preProcValues <- preProcess(training, method = "range")
trainTransformed <- predict(preProcValues, training)
testTransformed <- predict(preProcValues, testing)

testTransformed$dx <- as.factor(ifelse(testTransformed$dx == "cancer", 1, 0))
trainTransformed$dx <- as.factor(ifelse(trainTransformed$dx == "cancer", 1, 0))

grid <- expand.grid(cost = c(0.08, 0.09, 0.092, 0.094, 0.096, 0.098, 0.1),
                    Loss = c("L1", "L2"))
mod <- train(dx ~ .,
             data=trainTransformed, 
             method = svm,
             metric = "Accuracy",
             trControl = trainControl(method = "repeatedcv", number = 5, repeats = 10), tuneGrid = grid)

cv_acc <- getTrainPerf(mod)$TrainAccuracy
#predicting on the training dataset
result_train<-as.data.frame(predict(mod,trainTransformed))   
dataframe<-data.frame(result_train$`predict(mod, trainTransformed)`,trainTransformed$dx)
colnames(dataframe)<-c("x","y")
# training a logistic regression model on the cross validation dataset
model_log<-glm(y~x,data = dataframe,family = binomial)
# Predicting on the test dataset using Platt Scaling
result_test<-as.data.frame(predict(mod,newdata = testTransformed))
dataframe1<-data.frame(result_test$`predict(mod, newdata = testTransformed)`)
colnames(dataframe1)<-c("x")
result_test_platt<-predict(model_log,dataframe1,type="response")
test_roc <- roc(testTransformed$dx, 
                result_test_platt)
test_auc <- test_roc$auc
