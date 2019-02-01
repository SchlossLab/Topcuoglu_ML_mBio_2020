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
#    Filename to put to function: 
#     "Random_Forest"


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
                     repeats = 10,
                     number=5,
                     returnResamp="final",
                     classProbs=TRUE,
                     summaryFunction=twoClassSummary,
                     indexFinal=NULL,
                     savePredictions = TRUE)
  # Grid and caret method defined for random forest classification model
  if(model=="Random_Forest"){
    grid <-  expand.grid(mtry = c(80,500,1000,1500))
    method = "rf"
  }
  else { 
    print("Model not available")
  }
  params <- list(grid, method, cv)
  return(params)
}

