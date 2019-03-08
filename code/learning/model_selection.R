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
# -------------------------CHANGED--------------------------------------->  
# ADDED cv index to make sure the internal 5-folds are stratified for diagnosis. 
  folds <- 5
  cvIndex <- createMultiFolds(factor(train_data$dx), folds, times=10)
  cv <- trainControl(method="repeatedcv",
                     repeats = 100,
                     number=folds,
                     index = cvIndex,
                     returnResamp="final",
                     classProbs=TRUE,
                     summaryFunction=twoClassSummary,
                     indexFinal=NULL,
                     savePredictions = TRUE)
# # -----------------------------------------------------------------------> 
  #################################################################################  
# For linear models we are using LiblineaR package
#
# LiblineaR can produce 10 types of (generalized) linear models:
# The regularization can be 
#     1. L1 
#     2. L2
# The losses can be:
#     1. Regular L2-loss for SVM (hinge loss), 
#     2. L1-loss for SVM
#     3. Logistic loss for logistic regression. 
#
# Here we will use L1 and L2 regularization and hinge loss (L2-loss) for linear SVMs
# We will use logistic loss for L2-resularized logistic regression
# The liblinear 'type' choioces are below:
#
# for classification
# • 0 – L2-regularized logistic regression (primal)---> we use this for l2-logistic
#  • 1 – L2-regularized L2-loss support vector classification (dual)
#  • 2 – L2-regularized L2-loss support vector classification (primal) ---> we use this for l2-linear SVM
#  • 3 – L2-regularized L1-loss support vector classification (dual)
#  • 4 – support vector classification by Crammer and Singer
#  • 5 – L1-regularized L2-loss support vector classification---> we use this for l1-linear SVM
#  • 6 – L1-regularized logistic regression
#  • 7 – L2-regularized logistic regression (dual)
#  
#for regression 
#  • 11 – L2-regularized L2-loss support vector regression (primal)
#  • 12 – L2-regularized L2-loss support vector regression (dual)
#  • 13 – L2-regularized L1-loss support vector regression (dual)
  #################################################################################
  
  
  #################################################################################
  # We use ROC metric for all the models
  # To do that I had to make changes to the caret package functions.
  # The files 'data/caret_models/svmLinear3.R and svmLinear5.R are my functions. 
  # I added 1 line to get Decision Values for linear SVMs:
  #
  #           prob = function(modelFit, newdata, submodels = NULL){
  #             predict(modelFit, newdata, decisionValues = TRUE)$decisionValues
  #           },
  #
  # This line gives decision values instead of probabilities and computes ROC in:
  #   1. train function with the cross-validataion
  #   2. final trained model
  # using decision values and saves them in the variable "prob"
  # This allows us to pass the cv function for all models:
  # cv <- trainControl(method="repeatedcv",
  #                   repeats = 100,
  #                   number=folds,
  #                   index = cvIndex,
  #                   returnResamp="final",
  #                   classProbs=TRUE,
  #                   summaryFunction=twoClassSummary,
  #                   indexFinal=NULL,
  #                   savePredictions = TRUE)
  # 
  # There parameters we pass for L1 and L2 SVM:
  #                   classProbs=TRUE,
  #                   summaryFunction=twoClassSummary,
  # are actually computing ROC from decision values not probabilities
  #################################################################################
  
  # Grid and caret method defined for each classification models
  if(model=="L2_Logistic_Regression") {
    grid <-  expand.grid(cost = c(0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 10),
                         loss = "L2_primal", 
                         # This chooses type=0 for liblinear R package 
                         # which is logistic loss, primal solve for L2 regularized logistic regression
                         epsilon = 0.01) #default epsilon recommended from liblinear
    method <- "regLogistic"
  }
  else if (model=="L1_Linear_SVM"){ # 
    grid <- expand.grid(cost = c(0.0001, 0.001, 0.01, 0.015, 0.025, 0.05, 0.1, 0.5, 1),
                        Loss = "L2")
    method <- "svmLinear5" # I wrote this function in caret
  }
  else if (model=="L2_Linear_SVM"){ 
    grid <- expand.grid(cost = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1),
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
