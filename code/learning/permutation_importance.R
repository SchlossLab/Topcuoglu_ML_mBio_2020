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

  # -----------Get the original testAUC from held-out test data--------->
  # Calculate the test-auc for the actual pre-processed held-out data
  rpartProbs <- predict(model, full, type="prob")
  base_roc <- roc(ifelse(full$dx == "cancer", 1, 0), rpartProbs[[1]])
  base_auc <- base_roc$auc
  # -------------------------------------------------------------------->

  # ----------- Read in the correlation matrix of full dataset---------->
  # Get the correlation matrix made by full dataset
  # This correlation matrix used Spearman correlation
  # Only has the correlatons that has:
  #     1. Coefficient = 1
  #     2. Adjusted p-value < 0.01
  corr <- read_csv("data/process/sig_flat_corr_matrix.csv") %>%
    select(-p, -cor)
  # -------------------------------------------------------------------->

  # ----------- Get the names of correlated OTUs------------------------>
  # Get the correlated unique OTU ids
  correlated_otus <- unique(c(corr$row, corr$column))
  # -------------------------------------------------------------------->

  # ----------- Get the names of non-correlated OTUs-------------------->
  # Remove those names as columns from full test data
  # Remove the diagnosis column to only keep non-correlated features
  non_correlated_otus <- full %>%
    select(-correlated_otus) %>%
    select(-dx) %>%
    colnames()
  # -------------------------------------------------------------------->

  # ----------- Get feature importance of non-correlated OTUs------------>
  # Start the timer
  library(tictoc)
  tic("perm")
  # Permutate each feature in the non-correlated dimensional feature vector
  # Here we are
  #     1. Permuting the values in the OTU column randomly for each OTU in the list
  #     2. Applying the trained model to the new test-data where 1 OTU is randomly shuffled
  #     3. Getting the new AUROC value
  #     4. Calculating how much different the new AUROC is from original AUROC
  # Because we do this with lapply we randomly permute each OTU one by one.
  # We get the impact each non-correlated OTU makes in the prediction performance (AUROC)
  non_corr_imp <- do.call('rbind', lapply(non_correlated_otus, function(i){
    full_permuted <- full
    full_permuted[,i] <- sample(full[,i])
    # Predict the diagnosis outcome with the one-feature-permuted test dataset
    rpartProbs_permuted <- predict(model, full_permuted, type="prob")
    # Calculate the new auc
    new_auc <- roc(ifelse(full_permuted$dx == "cancer", 1, 0), rpartProbs_permuted[[1]])$auc
    # Return how does this feature being permuted effect the auc
    return(new_auc)
  }))
  print(non_corr_imp)
  # Save non correlated results in a dataframe.
  non_corr_imp <- as.data.frame(non_corr_imp) %>%
    mutate(names=factor(non_correlated_otus)) %>%
    rename(new_auc=V1)
  # -------------------------------------------------------------------->


  # ----------- Get feature importance of correlated OTUs -------------->

  # Have each OTU in a group with all the other OTUs its correlated with
  # Each OTU should only be in a group once.
  non_matched_corr <- corr %>% filter(!row %in% column) %>%
    group_by(row)


  # Use that tidyverse grouping to get a list of the OTUs that are grouped
  # Turn it into a character list of the OTU names with their correlated OTUs
  # 432 is the number of groups of correlated OTUs
  split <- group_split(non_matched_corr)
  groups <- lapply(1:432, function(i){
  grouped_corr_otus <- split[[i]][2] %>%
    add_case(column=unlist(unique(split[[i]][1])))
  return(grouped_corr_otus)
  })
  groups_list <- map(groups[1:432], "column")
  groups_list_sorted <- map(groups_list[1:432], sort)


  # Permute the grouped OTUs together and calculate AUC change
  corr_imp <- do.call('rbind', lapply(groups_list_sorted, function(i){
    full_permuted_corr <- full
    full_permuted_corr[,unlist(groups_list_sorted[i])] <- sample(full[,unlist(groups_list_sorted[i])])
    # Predict the diagnosis outcome with the one-feature-permuted test dataset
    rpartProbs_permuted_corr <- predict(model, full_permuted_corr, type="prob")
    # Calculate the new auc
    new_auc <- roc(ifelse(full_permuted_corr$dx == "cancer", 1, 0), rpartProbs_permuted_corr[[1]])$auc
    list <- list(new_auc, unlist(i))
    return(list)
  }))
  print(corr_imp)

  # save non correlated results in a dataframe
  # Create a bunch of columns so that each OTU in the group has its own column
  # We use seperate function to break up the grouped list otf OTUs
  # Now correlated OTUs are in one row, seperated by each OTU as columns
  # Last column has the percent AUC change per group of OTUs
  x <- as.character(seq(0, 432, 1))
  corr_imp_appended <- as.data.frame(corr_imp) %>%
    separate(V2, into = x)
  # Unlist percent auc change to save it as a csv later
  results <- corr_imp_appended %>%
    mutate(new_auc=unlist(corr_imp_appended$V1))
  # Only keep the columns that are not all NA
  not_all_na <- function(x) any(!is.na(x))
  correlated_auc_results <- results %>%
    select(-V1, -"0") %>%
    select_if(not_all_na)
  # -------------------------------------------------------------------->


  # stop timer
  secs <- toc()
  walltime <- secs$toc-secs$tic
  print(walltime)
  # Save the original AUC, non-correlated importances and correlated importances
  roc_results <- list(base_auc, non_corr_imp, correlated_auc_results)
  return(roc_results)
}
