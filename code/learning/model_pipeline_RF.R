# Author: Begum Topcuoglu
# Date: 2019-01-14
######################################################################
# Description:
# This script trains and tests the model according to proper pipeline
######################################################################

######################################################################
# Dependencies and Outputs: 
#    Model to put to function: 
#     "Random_Forest"


#    Dataset to put to function: 
#         Features: Hemoglobin levels and 16S rRNA gene sequences in the stool 
#         Labels: - Colorectal lesions of 490 patients. 
#                 - Defined as cancer or not.(Cancer here means: SRN)
# 
# Usage:
# Call as source when using the function. The function is:
#   pipeline(data, model) 

# Output:
#  List of:
#     1. AUCs  for cv of 100 data-splits
#     2. AUCS for test of 100 data-splits
######################################################################

######################################################################
#------------------------- DEFINE FUNCTION -------------------#
######################################################################
pipeline <- function(dataset, model){
  # Create vectors to save cv and test AUC values for every data-split
  results_total <-  data.frame()
  test_aucs <- c()
  cv_aucs <- c()
  # Do the 80-20 data-split 
  
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
  if(model=="Random_Forest"){
    print(model)
    trained_model <-  train(dx ~ .,
                            data=trainTransformed,
                            method = method,
                            trControl = cv,
                            metric = "ROC",
                            tuneGrid = grid,
                            ntree=1000)
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
  # Save all the cv meanAUCs over iterations in cv_aucs
  cv_aucs <- c(cv_aucs, cv_auc)
  # Save all results of hyper-parameters and their corresponding meanAUCs for each iteration
  results_individual <- trained_model$results
  results_total <- rbind(results_total, results_individual)
  
  results <- list(cv_aucs, test_aucs, results_total)
  return(results)
}
