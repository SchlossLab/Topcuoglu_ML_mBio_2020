# Author: Begum Topcuoglu
# Date: 2018-06-05
#
######################################################################
# This script takes 100 data/temp/*cor_results_*LINEAR_MODELS*
# It takes the abolute values of weights and ranks OTUs based on weights
# Saves the ranking information for all of them as individual 100 .csv files
# To merge them, use code/merge_feature_ranks.sh
# That will create a combined tsv file for each linear model with 100 datasplits
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

source("code/learning/functions.R")



create_feature_rankings <- function(data, model_name){
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
    #     b) Rank them based on that hierarchy
    #     c) Put the signs back to weights
    #     d) select the OTU names, weights and ranks, return dataframe
    ranks <- weights %>%
      arrange(desc(value)) %>%
      mutate(rank = 1:nrow(weights)) %>%
      mutate(value = case_when(sign=="negative" ~ value*-1,
                              sign=="positive"~ value,
                              sign=="zero" ~ value)) %>%
      select(key, value, rank, sign)

  return(ranks)
}


######################################################################
#--------------Run the functions and create ranking files ----------#
######################################################################

# ----------- Read in saved weights for linear models in temp folder ---------->
# List the files with feature weights with the pattern that has an "L" which only selects linear models.
# Correlated files for linear models has the weights for OTUs in trained model:
cor_files <- list.files(path= 'data/temp', pattern='all_imp_features_cor_results_L.*', full.names = TRUE)

# Take each file 1-100 and add ranks as a column to it.
# Then save that file with the "_feature_ranking_#" extension
# These will be then merged in shell bash
i <- 0
for(file_name in cor_files){
  i <- i + 1
  importance_data <- read_files(file_name)
  model_name <- as.character(importance_data$model[1])# get the model name from table
create_feature_rankings(importance_data, model_name) %>%
    as.data.frame() %>%
    write_tsv(., paste0("data/temp/", model_name, "_feature_ranking_", i, ".tsv"))
}
