# Author: Begum Topcuoglu
# Date: 2018-02-13
#
######################################################################
# This script looks at the model interpretation
######################################################################


######################################################################
#----------------- Read in necessary libraries -------------------#
######################################################################
deps = c("reshape2", "cowplot", "ggplot2","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}

# Load in needed functions and libraries
source('code/learning/functions.R')

######################################################################
# Load .tsv data generated with modeling pipelines
######################################################################

# Read in the important features files
interp_files <- list.files(path= 'data/process', pattern='combined_all_imp.*', full.names = TRUE)

read_files <- function(filenames){
  for(file in filenames){
    # Read the files generated in main.R 
    # These files have cvAUCs and testAUCs for 100 data-splits
    data <- read.delim(file, header=T, sep=',')
  }
  return(data)
}

get_interp_info <- function(model, model_name){ 
  if(model_name=="L2_Logistic_Regression" || 
     model_name=="L1_Linear_SVM" || 
     model_name=="L2_Linear_SVM")
      {
      ranked_imp <- model %>% 
      select(-normal) %>% 
      group_by(names) %>% 
      summarise(mean_imp = mean(cancer), sd_imp = sd(cancer), n = n()) %>% 
      arrange(-n) %>% 
      head(n=10)
      }
  else{
      ranked_imp <- model %>% 
      group_by(names) %>% 
      summarise(mean_imp = mean(Overall), sd_imp = sd(Overall), n = n()) %>% 
      arrange(-n) %>% 
      head(n=10)
      }
  return(ranked_imp)
}

for(file_name in interp_files){
  importance_data <- read_files(file_name)
  model_name <- as.character(importance_data$model[1]) # get the model name from table
  get_interp_info(importance_data, model_name) %>% 
  write_tsv(., paste0("data/process/", model_name, "_importance.tsv"))
}
    


