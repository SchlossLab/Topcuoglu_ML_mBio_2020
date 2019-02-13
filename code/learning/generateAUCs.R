######################################################################
# Author: Begum Topcuoglu
# Date: 2019-01-15
# Title: Generate files that has cv and test AUCs for 1 data-split 
######################################################################

######################################################################
# Dependencies and Outputs: 
# This function accept:
#   1. Data file generated in main.R
#   2. Model name defined in command line: 
#       "L2_Logistic_Regression"
#       "L1_Linear_SVM"
#       "L2_Linear_SVM"
#       "RBF_SVM"
#       "Decision_Tree"
#       "Random_Forest"
#       "XGBoost"
#   3. Seed number defined in command line:
#       [1-100]


# Call as source when using the function. The function is:
#   get_AUCs()

# The output:
#  Results .csv files:
#     1. cvAUC and testAUC for 1 data-split
#     2. cvAUC for all hyper-parameters during tuning for 1 datasplit
#     3. feature importance info on first 10 features for 1 datasplit
######################################################################


######################################################################
#------------------------- DEFINE FUNCTION -------------------#
######################################################################
get_results <- function(dataset, models, split_number){
  for(ml in models){
  
  # Save results of the modeling pipeline as a list
  results <- pipeline(dataset, ml) 
  
  # ------------------------------------------------------------------ 
  # Create a matrix with cv_aucs and test_aucs from 1 data split
  aucs <- matrix(c(results[[1]], results[[2]]), ncol=2) 
  # Convert to dataframe and add a column noting the model name
  aucs_dataframe <- data.frame(aucs) %>% 
    rename(cv_aucs=X1, test_aucs=X2) %>% 
    mutate(model=ml) %>% 
    write.csv(file=paste0("data/temp/best_hp_results_", ml,"_", split_number, ".csv"), row.names=F)
  # ------------------------------------------------------------------   

  # ------------------------------------------------------------------   
  # Save training results for 1 datasplit and corresponding AUCs
  all_results <- results[3]
  # Convert to dataframe and add a column noting the model name
  dataframe <- data.frame(all_results) %>% 
    mutate(model=ml) %>% 
    write.csv(file=paste0("data/temp/all_hp_results_", ml,"_", split_number, ".csv"), row.names=F)
  # ------------------------------------------------------------------ 
  
  # ------------------------------------------------------------------   
  # Save 10 feature importance of the model for 1 datasplit
  imp_features <- results[4]
  # Convert to dataframe and add a column noting the model name
  dataframe <- data.frame(imp_features) %>% 
    mutate(model=ml) %>% 
    write.csv(file=paste0("data/temp/all_imp_features_results_", ml,"_", split_number, ".csv"), row.names=F)
  # ------------------------------------------------------------------ 
  }
}

