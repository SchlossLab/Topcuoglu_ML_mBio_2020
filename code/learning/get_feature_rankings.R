# Author: Begum Topcuoglu
# Date: 2018-02-13
#
######################################################################
# This script takes data/temp/*cor_results_*LINEAR_MODELS*
# It takes the abolute values of all and sorts them from large to small
# Saves top 20 as .csv file
######################################################################


######################################################################
#----------------- Read in necessary libraries -------------------#
######################################################################
deps = c("cowplot","reshape2", "cowplot", "ggplot2","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}
######################################################################
#----------------- Define the functions we will use -----------------#
######################################################################


# -------------------- Read files ------------------------------------>
# This function:
#     1. takes a list of files(with their path)
#     2. reads them as delim files with comma seperator
#     3. returns the dataframe
read_files <- function(filenames){
  for(file in filenames){
    # Read the files generated in main.R 
    # These files have cvAUCs and testAUCs for 100 data-splits
    data <- read.delim(file, header=T, sep=',')
  }
  return(data)
}
# -------------------------------------------------------------------->

# ----------- Read in saved weights for linear models in temp folder ---------->
# List the important features files by defining a pattern in the path
# Correlated files are:
cor_files <- list.files(path= 'data/temp', pattern='all_imp_features_cor_results_L.*', full.names = TRUE)

get_interp_info <- function(data, model_name){ 
    # If the models are linear, we saved the weights of every OTU for each datasplit
    # We want to plot the ranking of OTUs for linear models. 
    # 1. Get dataframe transformed into long form
    #         The OTU names are in 1 column
    #         The weight value are in 1 column
    weights <- data %>% 
      select(-Bias, -model) %>% 
      gather(factor_key=TRUE) %>% 
      mutate(sign = case_when(value<0 ~ "negative",
                              value>0 ~ "positive",
                              value==0 ~ "zero")) 
    
    # 2. We change all the weights to their absolute value
    #       Because we want to see which weights are the largest 
    weights$value <- abs(weights$value)
    # 3.  a) Order the dataframe from largest weights to smallest.
    #     b) Select the largest 10 
    #     c) Put the signs back to weights
    #     d) select the OTU names, mean weights with their signs and the sd
    ranks <- weights %>% 
      arrange(desc(value)) %>% 
      mutate(rank = 1:nrow(weights)) %>% 
      mutate(value = case_when(sign=="negative" ~ value*-1,
                              sign=="positive"~ value, 
                              sign=="zero" ~ value)) %>% 
      select(key, value, rank)

  return(ranks)
}

i <- 0
for(file_name in cor_files){
  i <- i + 1
  importance_data <- read_files(file_name)
  model_name <- as.character(importance_data$model[1])# get the model name from table
  get_interp_info(importance_data, model_name) %>% 
    as.data.frame() %>% 
    write_tsv(., paste0("data/temp/", model_name, "_feature_ranking_", i, ".tsv"))
}
    
