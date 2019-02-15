
# Author: Begum Topcuoglu
# Date: 2019-01-14
######################################################################
# Description:
# This script trains and tests the model according to proper pipeline
######################################################################

######################################################################
# Dependencies and Outputs:
#    Model to put to function:
#       1. "L2_Logistic_Regression"
#       2. "L2_Linear_SVM"
#       3. "RBF_SVM"
#       4. "Decision_Tree"
#       5. "Random_Forest"
#       6. "XGBoost"
#    Dataset to put to function:
#         Features: Hemoglobin levels and 16S rRNA gene sequences in the stool
#         Labels: - Colorectal lesions of 490 patients.
#                 - Defined as cancer or not.(Cancer here means: SRN)
#
# Usage:
# Call as source when using the function. The function is:
#   pipeline(data, model)

# Output:
#  A results list of:
#     1. cvAUC and testAUC for 1 data-split
#     2. cvAUC for all hyper-parameters during tuning for 1 datasplit
#     3. feature importance info on first 10 features for 1 datasplit
#     4. trained model as a caret object
######################################################################

######################################################################
#------------------------- DEFINE FUNCTION -------------------#
######################################################################
pipeline <- function(dataset, model){

  # Create vectors to save results of pipeline
  results_total <-  data.frame()
  test_aucs <- c()
  cv_aucs <- c()
  
  # Do the 80-20 data-split
    # Stratified data partitioning %80 training - %20 testing
    inTraining <- createDataPartition(dataset$dx, p = .80, list = FALSE)
    training <- dataset[ inTraining,]
    testing  <- dataset[-inTraining,]
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
      trained_model <-  train(dx ~ ., # label 
                              data=trainTransformed, #total data
                              method = method, 
                              trControl = cv,
                              metric = "ROC",
                              tuneGrid = grid,
                              family = "binomial")
    }
    if(model=="L1_Linear_SVM"){
      print(model)
      trained_model <-  train(dx ~ .,
                              data=trainTransformed,
                              method = method,
                              trControl = cv,
                              metric = "Accuracy", # not ROC due to package problems
                              tuneGrid = grid)
    }
    else if(model=="Random_Forest"){
      print(model)
      trained_model <-  train(dx ~ .,
                              data=trainTransformed,
                              method = method,
                              trControl = cv,
                              metric = "ROC",
                              tuneGrid = grid,
                              ntree=1000) # not tuning ntree
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
    if(model=="L1_Linear_SVM"){ # exception due to package problems for ROC calculation
      #################################################################################
      # We have to calculate ROC ourselves for cross-validation
      # I have made changes to the caret package functions to get Decision Values
      # In SVM we don't get predicted probabilities but decision values
      # I changed the function in caret to give us decision values instead of probabilities.
      #################################################################################
      # For cross-validation
        # selected indices are taking the best performing hyper-parameter only
        selectedIndices <-trained_model$pred[,6] == trained_model$bestTune[,1]
        # The repsonse is the known labels 
        cv.response <- trained_model$pred[selectedIndices, ]$obs
        # The predictor is the Decision Values that we get from training
        cv.predictor <- trained_model$pred[selectedIndices, ]$cancer
        # We use pROC function to calculate ROC from decision values
        # roc function gives us auc values as well
        cv_roc <- roc(cv.response, cv.predictor, auc=TRUE)
        cv_auc <- cv_roc$auc
        cv_aucs <- c(cv_aucs, cv_auc)
      
      # For testing we can use the type="prob" to get decision values(becuase of my function)
        rpartProbs <- predict(trained_model, testTransformed, type="prob")
        test_roc <- roc(ifelse(testTransformed$dx == "cancer", 1, 0), rpartProbs[[2]])
        test_auc <- test_roc$auc
        # Save all the test AUCs over iterations in test_aucs
        test_aucs <- c(test_aucs, test_auc)
        # complete results
        results_individual <- trained_model$results
        results_total <- rbind(results_total, results_individual)
    }
    else{
      #################################################################################
      # For all the other models, ROC calculation is already included.
      # We follow caret instructions
      #################################################################################
        # Mean AUC value over repeats of the best cost parameter during training
        cv_auc <- getTrainPerf(trained_model)$TrainROC
        # Predict on the test set and get predicted probabilities
        rpartProbs <- predict(trained_model, testTransformed, type="prob")
        test_roc <- roc(ifelse(testTransformed$dx == "cancer", 1, 0), rpartProbs[[2]])
        test_auc <- test_roc$auc
        # Save all the test AUCs over iterations in test_aucs
        test_aucs <- c(test_aucs, test_auc)
        # Cross-validation mean AUC value
        # Save all the cv meanAUCs over iterations in cv_aucs
        cv_aucs <- c(cv_aucs, cv_auc)
        # Save all results of hyper-parameters and their corresponding meanAUCs for each iteration
        results_individual <- trained_model$results
        results_total <- rbind(results_total, results_individual)
    }
  # Here we look at the top 10 important features 
  feature_importance <- model_interpret(trained_model)
  # Return all the metrics
  results <- list(cv_aucs, test_aucs, results_total, feature_importance, trained_model)
  return(results)
}
