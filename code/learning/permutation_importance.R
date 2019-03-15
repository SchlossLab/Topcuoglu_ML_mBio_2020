######################################################################
# Author: Begum Topcuoglu
# Date: 2018-03-15
# Title: Permutation Importance for features in each model
######################################################################

######################################################################
# Description: 

# This script will read in data from Baxter et al. 2016
#     - 0.03 subsampled OTU dataset
#     - CRC metadata: SRN information



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
  # Calculate the test-auc for the 
  rpartProbs <- predict(model, full, type="prob")
  base_roc <- roc(ifelse(full$dx == "cancer", 1, 0), rpartProbs[[1]])
  base_auc <- base_roc$auc
  
  imp <- c()
  library(tictoc)
  tic("perm")
  for (i in 1:5){
    full_permuted <- full 
    full_permuted[,i] <- sample(full[,i])
    rpartProbs_permuted <- predict(model, full_permuted, type="prob")
    new_roc <- roc(ifelse(full_permuted$dx == "cancer", 1, 0), rpartProbs_permuted[[1]])
    new_auc <- new_roc$auc
    percent_increase_error=((base_auc-new_auc)/base_auc)*100
    imp <- c(imp, percent_increase_error)
  }
  secs <- toc()
  walltime <- secs$toc-secs$tic
  print(walltime)
  roc_results <- c(base_auc, new_auc, imp)
  return(roc_results)
}



