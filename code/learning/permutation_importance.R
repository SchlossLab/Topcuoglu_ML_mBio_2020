######################################################################
# Author: Begum Topcuoglu
# Date: 2018-03-15
# Title: Permutation Importance for features in each model
######################################################################

######################################################################
# Description: 

# This script will read in:
#     - Trained model
#     - Pre-processed hel-out test data



# It will run the following:
#     - Predict the held-out test data for one data-split
#     - Permutate each feature in the test data randomly
#     - Predict transformed test data for each permutation
#     - Substract transformed prediction auc from original prediction auc 
#     - Determine which feature makes the biggest difference in auc
######################################################################

######################################################################
# Dependencies and Outputs: 

#     - The funtion needs transformed test set
#     - Trained model for one data-split

# Be in the project directory.

# The outputs are:
#   (1) AUC difference for each feature transformation 
######################################################################


################### IMPORT LIBRARIES and FUNCTIONS ###################
# The dependinces for this script are consolidated in the first part
deps = c("tictoc", "caret", "pROC", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), 
                     quiet=TRUE, 
                     repos = "http://cran.us.r-project.org", dependencies=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}
######################################################################



####################### DEFINE FUNCTION  #############################
permutation_importance <- function(model, full){
  # Calculate the test-auc for the actual pre-processed held-out data
  rpartProbs <- predict(model, full, type="prob")
  base_roc <- roc(ifelse(full$dx == "cancer", 1, 0), rpartProbs[[1]])
  base_auc <- base_roc$auc
  
  # Start the timer
  library(tictoc)
  tic("perm")
  # Permutate each feature in a 6921 dimensional feature vector
  imp <- do.call('rbind', lapply(1:6921, function(i){    
    full_permuted <- full 
    full_permuted[,i] <- sample(full[,i])
    # Predict the diagnosis outcome with the one-feature-permuted test dataset
    rpartProbs_permuted <- predict(model, full_permuted, type="prob")
    # Calculate the new auc
    new_auc <- roc(ifelse(full_permuted$dx == "cancer", 1, 0), rpartProbs_permuted[[1]])$auc
    # Return how does this feature being permuted effect the auc
    return(((base_auc-new_auc)/base_auc)*100)
  }))
  # stop timer
  secs <- toc()
  walltime <- secs$toc-secs$tic
  print(walltime)
  # save results
  imp <- as.data.frame(imp) %>% 
    mutate(names=colnames(full[,1:6921])) %>% 
    rename(percent_auc_change=V1)
  roc_results <- list(base_auc, imp)
  return(roc_results)
}



