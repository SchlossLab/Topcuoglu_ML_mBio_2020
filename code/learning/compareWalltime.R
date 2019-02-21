# Author: Begum Topcuoglu
# Date: 2018-02-12
#
######################################################################
# This script plots Figure 1:
#   1. cvAUC (means of 100 repeats for the best hp) of 100 datasplits
#   2. testAUC of 100 datasplits
######################################################################

######################################################################
# Load in needed functions and libraries
source('code/learning/functions.R')
# detach("package:randomForest", unload=TRUE) to run
######################################################################

# Load .csv data generated with modeling pipeline 
######################################################################

# Read in the cvAUCs, testAUCs for 100 splits.
walltime_files <- list.files(path= 'data/process', pattern='walltime*', full.names = TRUE)

logit <- read_files(walltime_files[1]) %>% 
  summarise_walltime(walltime_files[1])
  

            