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
inTraining <- createDataPartition(data$dx, p = .80, list = FALSE)
training <- data[ inTraining,]
testing  <- data[-inTraining,]
# Scale all features between 0-1
preProcValues <- preProcess(training, method = "range")
trainTransformed <- predict(preProcValues, training)
x_train <- trainTransformed %>% 
  select(-dx)
y_train <- trainTransformed %>% 
  select(dx)
testTransformed <- predict(preProcValues, testing)
x_test <- testTransformed %>% 
  select(-dx)
y_test <- testTransformed %>% 
  select(dx)

#########################################################################
model <- LiblineaR(x_train, trainTransformed$dx, type = 5)
pred=predict(model,x_test, decisionValues = TRUE)
# https://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf
deci <- pred[[2]]
label <- ifelse(y_test == "cancer", TRUE, FALSE)
prior1 <- sum(label==TRUE)
prior0 <- sum(label==FALSE)
#Parameter setting
maxiter <- 100 # Maximum number of iterations
minstep <-1e-10 # Minimum step taken in line search
sigma <- 1e-12 # Set to any value > 0
#Construct initial values: target support in array t,
# initial function value in fval
hiTarget <- (prior1+1.0)/(prior1+2.0)
loTarget <- 1/(prior0+2.0)
len=prior1+prior0 # Total number of data
t <- c()
for(i in 1:length(len)){
  print(i)
  if (label[i] > 0)
    t[i] <- hiTarget
  else
    t[i] <- loTarget
}
