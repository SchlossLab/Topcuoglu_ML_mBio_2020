######################################################################
# Author: Begum Topcuoglu
# Date: 2019-01-15
# Title: Generate files that has cv and test AUCs for100 data-split 
######################################################################

######################################################################
# Dependencies and Outputs: 
# This function accept files generated in main.R
#    Filenames to put to function: 
#       1. "L2_Logistic_Regression"
#       2. "L2_Linear_SVM"
#       3. "RBF_SVM"
#       4. "Decision_Tree"
#       5. "Random_Forest"
#       6. "XGBoost"


# Call as source when using the function. The function is:
#   get_AUCs()

# The output:
#  A results .csv file with:
#     1. AUCs  for cv of 100 data-splits
#     2. AUCS for test of 100 data-splits
######################################################################

######################################################################
#----------------- Read in necessary libraries -------------------#
######################################################################

deps = c("reshape2", "kernlab","LiblineaR", "doParallel","pROC", "caret", "gtools", "tidyverse", "ggpubr", "ggplot2","knitr","rmarkdown","vegan");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}
# Load in needed functions and libraries
source('code/learning/functions.R')


######################################################################
#------------------------- DEFINE FUNCTION -------------------#
######################################################################
get_AUCs <- function(models, split_number){
  for(ml in models){
  
  # Save results of the modeling pipeline as a list
  results <- pipeline(data, ml) 
  
  # ------------------------------------------------------------------ 
  # Create a matrix with cv_aucs and test_aucs from 100 data splits
  aucs <- matrix(c(results[[1]], results[[2]]), ncol=2) 
  # Convert to dataframe and add a column noting the model name
  aucs_dataframe <- data.frame(aucs) %>% 
    rename(cv_aucs=X1, test_aucs=X2) %>% 
    mutate(model=ml) %>% 
    write.csv(file=paste0("data/temp/best_hp_results_", ml,"_", split_number, ".csv"), row.names=F)
  # ------------------------------------------------------------------   

  # ------------------------------------------------------------------   
  # Save all tunes from 100 data splits and corresponding AUCs
  all_results <- results[3]
  # Convert to dataframe and add a column noting the model name
  dataframe <- data.frame(all_results) %>% 
    mutate(model=ml) %>% 
    write.csv(file=paste0("data/temp/all_hp_results_", ml,"_", split_number, ".csv"), row.names=F)
  # ------------------------------------------------------------------ 
  }
}

