# Author: Begum Topcuoglu
# Date: 2018-02-13
#
######################################################################
# This script looks at the model interpretation
######################################################################


######################################################################
#----------------- Read in necessary libraries -------------------#
######################################################################
deps = c("matrixStats","reshape2", "cowplot", "ggplot2","knitr","rmarkdown","vegan","gtools", "tidyverse");
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

get_interp_info <- function(data, model_name){ 
  if(model_name=="L2_Logistic_Regression" || 
     model_name=="L1_Linear_SVM" || 
     model_name=="L2_Linear_SVM"){
    # If the models are linear, we saved the weights of every OTU in the linear model for each datasplit
      # 1. Get the mean of weigths for each OTU (OTUs are columns)
      weights <- data %>% 
        select(-Bias, -model) %>% 
        gather(factor_key=TRUE) %>% 
        group_by(key) %>% 
        summarise(mean_weights = mean(value), sd_weights = sd(value)) %>% 
      # 2. We now want to save to a new column the sign of the weights
        mutate(sign = case_when(mean_weights<0 ~ "negative",
                                mean_weights>0 ~ "positive",
                                mean_weights==0 ~ "zero")) 
      # 3. We change all the weights to their absolute value
      weights$mean_weights <- abs(weights$mean_weights)
      # 4.  a) Order the dataframe from largest weights to smallest.
      #     b) Select the largest 10 
      #     c) Put the signs back to weights
      imp_means <- weights %>% 
        arrange(desc(mean_weights)) %>% 
        head(n=10) %>% 
        mutate(mean_weights = case_when(sign=="negative" ~ mean_weights*-1,
                                  sign=="positive"~ mean_weights)) %>% 
        select(key, mean_weights, sd_weights)

  }
  else if(model_name=="RBF_SVM"){
    imp_means <- data %>% 
        select(-normal) %>% 
        group_by(names) %>% 
        summarise(mean_imp = mean(cancer), sd_imp = sd(cancer), n = n()) %>% 
        arrange(-n) %>% 
        head(n=10)
  }
  else{
    # If the models are not linear, we saved variable importance of the top 10 variables per each datasplit
      # We will group by the OTU names 
    imp_means <- data %>% 
      group_by(names) %>% 
      # We then get the mean of importance of each OTU and how many times that OTU was saved 
        # How many data-splits actually chose that OTU as important
      summarise(mean_imp = mean(Overall), sd_imp = sd(Overall), n = n()) %>% 
      # Order the dataframe by how many times the OTU was observed 
        # Choose the top 10
      arrange(-n) %>% 
      head(n=10) 
      }
  return(imp_means)
}

for(file_name in interp_files){
  importance_data <- read_files(file_name)
  model_name <- as.character(importance_data$model[1]) # get the model name from table
  get_interp_info(importance_data, model_name) %>% 
    as.data.frame() %>% 
    write_tsv(., paste0("data/process/", model_name, "_importance.tsv"))
}
    


