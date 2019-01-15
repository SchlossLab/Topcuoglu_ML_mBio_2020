######################################################################
# Author: Begum Topcuoglu
# Date: 2019-01-15
# Title: Function to get the stats of cv and test AUCs of different models 
######################################################################

######################################################################
# Description: 

# This function accept files generated in main.R
#    Example filenames to put to function: 
#     - "results_L2_Logistic_Regression.csv"
#     - "results_Random_Forest.csv"

# It can use the files for:
#     - L2 Logistic Regression 
#     - L1 and L2 Linear SVM
#     - RBF SVM
#     - Decision Tree
#     - Random Forest 
#     - XGBoost 
######################################################################

######################################################################
# Dependencies and Outputs: 

# Call as source when using the function. The function is:
#   get_stats()
#   Example filenames to put to function: 
#     - "results_L2_Logistic_Regression.csv"
#     - "results_Random_Forest.csv"

# The output:
#  A stat file with:
#     1. meanAUCs and stdAUCS for cv
#     2. meanAUCs and stdAUCS for test
#     3. Difference in means of cv and test
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
#---------------------- FUNCTION ---------------------------#
######################################################################

get_stats <- function(filenames){
  for(file in filenames){
    # Read the files generated in main.R 
    # These files have cvAUCs and testAUCs for 100 data-splits
    data <- read.delim(file=file, header=T, sep=',') %>% 
    melt_data() %>% # Make data tidy
    select(-variable) 
    # We want to summarise AUCs for grouped cv and test
    # We get the mean and standard deviation 
    summary <- data %>% 
    group_by(Performance) %>% 
    summarise(meanAUC=mean(AUC), std=sd(AUC)) 
    # Assign cv and test means to variables
    cv_meanAUC <- summary[1,2]
    cv_sdAUC <- summary[1,3]
    test_meanAUC <- summary[2,2]
    test_sdAUC <- summary[2,3]
    # Calculate the difference between meanAUCs of cv and test 
    mean_difference <- cv_meanAUC - test_meanAUC
    # Save these stats as a list and turn into a dataframe
    # Add a column to the dataframe to annotate model name
    stats <- list(cv_meanAUC, 
                  cv_sdAUC, 
                  test_meanAUC, 
                  test_sdAUC, 
                  mean_difference) %>%     
              data.frame() %>% 
              rename(cv_meanAUC=meanAUC, 
                  cv_std=std, 
                  test_meanAUC=meanAUC.1, 
                  test_std=std.1, 
                  mean_difference=meanAUC.2) %>% 
              mutate(model = gsub("results_", "", file)) %>% 
              mutate(model = gsub(".csv", "", model))
  }  
  return(stats) # Return the dataframe
}

#################### USAGE WITH R-GENERATED .csv FILES ################
# Example filenames to put to function: 
#     - "results_L2_Logistic_Regression.csv"
#     - "results_Random_Forest.csv"
######################################################################

filenames_list <- list.files(path= "/Users/Begum/Documents/DeepLearning/", pattern='results_.*')
get_stats(filenames_list)