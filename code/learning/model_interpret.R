# Author: Begum Topcuoglu
# Date: 2019-01-14
######################################################################
# Description:
# This function defines:
#     - Interpretation of classification models we develop
#     - Will provide the important features in each model

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
#   model_interpret(trained.model)

# Output:
#  List of:
#     1. 10 most important features and their percent importance
######################################################################
library(tidyverse)
library(dplyr)
######################################################################
#------------------------- DEFINE FUNCTION -------------------#
######################################################################


model_interpret <- function(trained.model){
  # Notes on varImp function in caret:
  
  #       1. Each predictor will have a separate variable importance for each class.
  #       2. The varImp function automatically scales [0-100].
  #       3. Using scale = FALSE avoids this normalization step.
  
  
  # Here we create a readable format of all the features and their importance for each class.
  # col_index will be ranked for the most important feature to least.
  col_index <- varImp(trained.model, scale = FALSE)$importance %>% 
    mutate(names=row.names(.))
  sorted_ten <- col_index[order(-col_index[,1]),]
  # We will select the most important 10 features from each data-split
  # We have the names and the importance for each class.
  ten_imp <- head(sorted_ten, n=10)
  return(ten_imp)
}



