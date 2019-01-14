# Author: Begum Topcuoglu
# Date: 2019-01-14
######################################################################
# This script trains and tests the model according to proper pipeline
######################################################################

# Classifiers the function will accept are:
# models = c("L2_Logistic_Regression", "L2_Linear_SVM", "RBF_SVM", "Decision_Tree", "Random_Forest","XGBoost")
######################################################################


pipeline <- function(data, model){
  # Create vectors to save cv and test AUC values for every data-split
  best.tunes <- c()
  test_aucs <- c()
  cv_aucs <- c()
  # Loop to do 100 80-20 data-splits 
  for (i in 1:100) {
    # Stratified data partitioning %80 training - %20 testing
    inTraining <- createDataPartition(data$dx, p = .80, list = FALSE)
    training <- data[ inTraining,]
    testing  <- data[-inTraining,]
    # Scale all features between 0-1
    preProcValues <- preProcess(training, method = "range")
    trainTransformed <- predict(preProcValues, training)
    testTransformed <- predict(preProcValues, testing)
    # Define hyper-parameter tuning grid and the training method 
    grid <- tuning_grid(model)[[1]] 
    method <- tuning_grid(model)[[2]] 
    cv <- tuning_grid(model)[[3]] 
    # Train the model
    if(model=="L2_Logistic_Regression"){
      print(model)
      trained_model <-  train(dx ~ .,
                              data=trainTransformed,
                              method = method,
                              trControl = cv,
                              metric = "ROC",
                              tuneGrid = grid,
                              family = "binomial")
    }
    else if(model=="Random_Forest"){
      print(model)
      trained_model <-  train(dx ~ .,
                              data=trainTransformed,
                              method = method,
                              trControl = cv,
                              metric = "ROC",
                              tuneGrid = grid,
                              ntree=1000)
    }
    else{
      print(model)
      trained_model <-  train(dx ~ .,
                              data=trainTransformed,
                              method = method,
                              trControl = cv,
                              metric = "ROC",
                              tuneGrid = grid)
    }
    # Mean AUC value over repeats of the best cost parameter during training
    cv_auc <- getTrainPerf(trained_model)$TrainROC
    # Predict on the test set and get predicted probabilities
    rpartProbs <- predict(trained_model, testTransformed, type="prob")
    test_roc <- roc(ifelse(testTransformed$dx == "cancer", 1, 0), 
                    rpartProbs[[2]])
    test_auc <- test_roc$auc
    # Save all the test AUCs over iterations in test_aucs
    test_aucs <- c(test_aucs, test_auc)
    # Cross-validation mean AUC value
    # Save all the cv AUCs over iterations in cv_aucs
    cv_aucs <- c(cv_aucs, cv_auc)
  }
  results <- list(cv_aucs, test_aucs)
  return(results)
}
