library(tidyverse)
source('code/learning/functions.R')
######################################################################
#--------- Load cv and test AUROCs of 7 models for 100 datasplits--------#
######################################################################

# Read in the cvAUCs, testAUCs for 100 splits.
best_files <- list.files(path= 'data/process', pattern='combined_best.*', full.names = TRUE)
all <- map_df(best_files, read_files) 

plot <- histogram_p_value(all, "L2_Logistic_Regression","Random_Forest")

#resampling method shows that Random forest is best
perm_p_value(all, "L2_Logistic_Regression","Random_Forest")

