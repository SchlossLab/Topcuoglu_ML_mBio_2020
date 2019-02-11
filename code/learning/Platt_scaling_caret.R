library(tidyverse)
library(caret)
library(pROC)

######################## DATA PREPARATION #############################
# Features: Hemoglobin levels and 16S rRNA gene sequences in the stool
# Labels: - Colorectal lesions of 490 patients.
#         - Defined as cancer or not.(Cancer here means: SRN)
# Read in metadata and select only sample Id and diagnosis columns
meta <- read.delim('data/metadata.tsv', header=T, sep='\t') %>%
  select(sample, Dx_Bin, fit_result)
# Read in OTU table and remove label and numOtus columns
shared <- read.delim('data/baxter.0.03.subsample.shared', header=T, sep='\t') %>%
  select(-label, -numOtus)
# Merge metadata and OTU table.
# Group advanced adenomas and cancers together as cancer and normal, high risk normal and non-advanced adenomas as normal
# Then remove the sample ID column
data <- inner_join(meta, shared, by=c("sample"="Group")) %>%
  mutate(dx = case_when(
    Dx_Bin== "Adenoma" ~ "normal",
    Dx_Bin== "Normal" ~ "normal",
    Dx_Bin== "High Risk Normal" ~ "normal",
    Dx_Bin== "adv Adenoma" ~ "cancer",
    Dx_Bin== "Cancer" ~ "cancer"
  )) %>%
  select(-sample, -Dx_Bin) %>%
  drop_na()
# We want the diagnosis column to a factor
data$dx <- factor(data$dx, labels=c("normal", "cancer"))
#########################################################################

########################## DEFINE MODEL ################################
svm <- getModelInfo("svmLinear5")[[1]]

inTraining <- createDataPartition(data$dx, p = .80, list = FALSE)
training <- data[ inTraining,]
testing  <- data[-inTraining,]
# Scale all features between 0-1
preProcValues <- preProcess(training, method = "range")
trainTransformed <- predict(preProcValues, training)
testTransformed <- predict(preProcValues, testing)

grid <- expand.grid(cost = c(0.08, 0.09, 0.092, 0.094, 0.096, 0.098, 0.1, 0.5),
                    Loss = c("L1", "L2"))
mod <- train(dx ~ .,
             data=trainTransformed,
             method = svm,
             metric = "Accuracy",
             trControl = trainControl(method = "repeatedcv", number = 5, repeats = 10), tuneGrid = grid)

cv_acc <- getTrainPerf(mod)$TrainAccuracy
#predicting on the training dataset
testTransformed$dx <- as.factor(ifelse(testTransformed$dx == "cancer", 1, 0))
trainTransformed$dx <- as.factor(ifelse(trainTransformed$dx == "cancer", 1, 0))
result_train<-as.data.frame(predict(mod,trainTransformed, type="decision"))
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
print(test_auc)
