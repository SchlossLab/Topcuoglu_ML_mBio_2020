deps = c("caret","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}
meta <- read.delim('data/metadata.tsv', header=T, sep='\t') %>% 
  select(sample, dx)
  
shared <- read.delim('data/baxter.0.03.subsample.shared', header=T, sep='\t') %>% 
   select(-label, -numOtus)

data <- inner_join(meta, shared, by=c("sample"="Group")) %>% 
  filter(dx != 'adenoma') %>% 
  select(-sample) 
  
data$dx <- factor(data$dx, labels=c("normal", "cancer"))
test_auc_list = list()
cv_auc_list = list()
for (i in 1:10) {
  set.seed(20081989)
  inTraining <- createDataPartition(data$dx, p = .80, list = FALSE)
  training <- data[ inTraining,] 
  testing  <- data[-inTraining,]
  x_train <- training %>% select(-dx)
  y_train <- training$dx 
  y_train <- as.factor(y_train)
  grid <-  expand.grid(alpha=1, lambda = seq(0.01, 0.1, length = 100))
  cv <- trainControl(method="repeatedcv",repeats = 10, number=5, returnResamp="all",classProbs=TRUE,summaryFunction=twoClassSummary, indexFinal=NULL, savePredictions = TRUE)
  L2LogicalRegression <- train(x_train, y_train, method = "glmnet", trControl = cv, metric = "ROC",  tuneGrid = grid, family = "binomial")
  cv_auc <- getTrainPerf(L2LogicalRegression)$TrainROC
  # best parameter
  print(L2LogicalRegression$bestTune)
  # plot parameter performane 
  #trellis.par.set(caretTheme())
  #plot(L2LogicalRegression)
  # Select a parameter setting
  selectedIndices <- L2LogicalRegression$pred$lambda == L2LogicalRegression$bestTune$lambda
  rpartProbs <- predict(L2LogicalRegression, testing, type="prob")
  test_roc <- roc(ifelse(testing$dx == "cancer", 1, 0), rpartProbs[[2]])
  test_auc <- test_roc$auc
  cv_auc_list[[i]] <- cv_auc
  test_auc_list[[i]] <- test_auc
}
test_auc_list = do.call(rbind, test_auc_list)
cv_auc_list = do.call(rbind, cv_auc_list)


