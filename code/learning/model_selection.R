# Author: Begum Topcuoglu
# Date: 2019-01-14
######################################################################
# Description:
# This function defines defines:
#     1. Tuning budget as a grid the classification methods chosen
#     2. Cross-validation method
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
tuning_grid <- function(model){
  
  # Cross-validation method
  cv <- trainControl(method="repeatedcv",
                     repeats = 100,
                     number=5,
                     returnResamp="final",
                     classProbs=TRUE,
                     summaryFunction=twoClassSummary,
                     indexFinal=NULL,
                     savePredictions = TRUE)
  # Grid and caret method defined for each classification models
  if(model=="L2_Logistic_Regression") {
    grid <-  expand.grid(cost = c(0.001, 0.01, 0.1, 0.5, 1),
                         loss = c("L2_dual", "L1","L2_primal"),
                         epsilon = c(0.001, 0.01, 0.1))
    method <- "regLogistic"
  }
  else if (model=="L2_Linear_SVM"){
    grid <- expand.grid(C = c(0.05, 0.1, 0.12, 0.15, 0.2, 0.3))
    method <- "svmLinear"
  }
  else if (model=="L1_Linear_SVM"){
    grid <- expand.grid(cost = c(0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1 ,1),
                        Loss = "L2")
    method <- "svmLinear5"
  }
  else if (model=="RBF_SVM"){
    grid <-  expand.grid(sigma = c(0.0000001, 0.000001, 0.00001, 0.0001),
                         C = c(0.0001, 0.001, 0.01, 0.1))
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
                         eta=c(0.005, 0.01, 0.05),
                         max_depth=8,
                         colsample_bytree= 0.8,
                         min_child_weight=1,
                         subsample=c(0.6,0.7,0.8))
    method <- "xgbTree"
  }
  else { 
    print("Model not available")
  }
  params <- list(grid, method, cv)
  return(params)
}

