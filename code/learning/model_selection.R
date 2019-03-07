# Author: Begum Topcuoglu
# Date: 2019-01-14
######################################################################
# Description:
# This function defines:
#     1. Tuning budget as a grid for the classification methods chosen
#     2. Cross-validation method (how many repeats and folds)
#     3. Caret name for the classification method chosen
######################################################################

######################################################################
# Dependencies and Outputs:
#    Filenames to put to function:
#       1. "L2_Logistic_Regression"
#       2. "L2_Linear_SVM"
#       3. "RBF_SVM"
#       4. "Decision_Tree"
#       5. "Random_Forest"
#       6. "XGBoost"

# Usage:
# Call as source when using the function. The function is:
#   tuning_grid()

# Output:
#  List of:
#     1. Tuning budget as a grid the classification methods chosen
#     2. Cross-validation method
#     3. Caret name for the classification method chosen
######################################################################


######################################################################
#------------------------- DEFINE FUNCTION -------------------#
######################################################################
tuning_grid <- function(train_data, model){

#   Cross-validation method
#       5-fold
#       100 internal repeats to pick the best hp
#       Train the model with final hp decision to use model to predict
#       Return 2class summary and save predictions to calculate cvROC
  
  folds <- 5
  cvIndex <- createFolds(factor(train_data$dx), folds, returnTrain = T)
  cv <- trainControl(method="repeatedcv",
                     repeats = 1,
                     number=folds,
                     index = cvIndex,
                     returnResamp="final",
                     classProbs=TRUE,
                     summaryFunction=twoClassSummary,
                     indexFinal=NULL,
                     savePredictions = TRUE)
  # Grid and caret method defined for each classification models
  if(model=="L2_Logistic_Regression") {
    grid <-  expand.grid(cost = c(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1),
                         loss = "L2_primal", # This chooses type=0 for liblinear R package which is logistic loss, primal solve for L2 regularized logistic regression
                         epsilon = 0.01) #default epsioln recommended from liblinear
    method <- "regLogistic"
  }
  else if (model=="L1_Linear_SVM"){ # Exception due to package
    # Because I made changes to the package function, we can't:
    #     1. Get class probabilities and 2class summary
    #     2. We won't get ROC scores from cv
    #
    # We will get accuracy instead
    cv <- trainControl(method="repeatedcv",
                       repeats = 100,
                       number=folds,
                       index = cvIndex,
                       returnResamp="final",
                       classProbs=TRUE,
                       indexFinal=NULL,
                       savePredictions = TRUE)
    grid <- expand.grid(cost = c(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2),
                        Loss = "L2")
    method <- "svmLinear5" # I wrote this function in caret
  }
  else if (model=="L2_Linear_SVM"){ # Exception due to package
    # Because I made changes to the package function, we can't:
    #     1. Get class probabilities and 2class summary
    #     2. We won't get ROC scores from cv
    #
    # We will get accuracy instead
    cv <- trainControl(method="repeatedcv",
                       repeats = 1,
                       number=folds,
                       index = cvIndex,
                       returnResamp="final",
                       classProbs=TRUE,
                       indexFinal=NULL,
                       savePredictions = TRUE)
    grid <- expand.grid(cost = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1),
                        Loss = "L2")
    method <- "svmLinear3" # I changed this function in caret
  }
  else if (model=="RBF_SVM"){
    grid <-  expand.grid(sigma = c(0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1),
                         C = c(0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10))
    method <-"svmRadial"
  }
  else if (model=="Decision_Tree"){
    grid <-  expand.grid(maxdepth = c(1,2,3,4,5,6))
    method <-"rpart2"
  }
  else if (model=="Random_Forest"){
    grid <-  expand.grid(mtry = c(80,500,1000,1500))
    method = "rf"
  }
  else if (model=="XGBoost"){
    grid <-  expand.grid(nrounds=500,
                         gamma=0,
                         eta=c(0.001, 0.005, 0.01, 0.015, 0.02),
                         max_depth=8,
                         colsample_bytree= 0.8,
                         min_child_weight=1,
                         subsample=c(0.4, 0.5, 0.6))
    method <- "xgbTree"
  }
  else {
    print("Model not available")
  }
  # Return:
  #     1. the hyper-parameter grid to tune
  #     2. the caret function to train with
  #     3, cv method
  params <- list(grid, method, cv)
  return(params)
}
