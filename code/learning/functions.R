# Author: Begum Topcuoglu
# Date: 2018-12-06
#
######################################################################
# Place to store useful functions that will be used repeatedly throughout
######################################################################

deps = c("reshape2", "cowplot", "ggplot2","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}

# -------------------- Make performance files tidy------------------>
# Instead of 2 columns with names cv_aucs and test_aucs
# We will have 1 column with name Performance that tells us if test or cv
melt_data <-  function(data) {
  data_melt <- data %>%
    melt(measure.vars=c('cv_aucs', 'test_aucs')) %>%
    rename(AUC=value) %>%
    mutate(Performance = case_when(variable == "cv_aucs" ~ 'cross-validation', variable == "test_aucs" ~ 'testing')) %>%
    group_by(Performance)
  return(data_melt)
}
# -------------------------------------------------------------------->


# -------------------- Read files ------------------------------------>
# Read in files as delim that are saved in a list with a pattern
read_files <- function(filenames){
  for(file in filenames){
    # Read the files generated in main.R
    # These files have cvAUCs and testAUCs for 100 data-splits
    data <- read.delim(file, header=T, sep=',')
  }
  return(data)
}
# -------------------------------------------------------------------->


# -------------------- Extract model name----------------------------->
# Get model name with sub from file name
get_model_name <- function(files){
  pat1 <- "data/process/walltime_"
  name_files <- sub(pat1, "", files)
  pat2 <- ".csv"
  names <- sub(pat2, "", name_files)
  return(names)
}
# -------------------------------------------------------------------->


# ------------------- Re-organize feature importance  ----------------->
# This function:
#     1. Takes in a dataframe (different data for each model) and the model name
#     2. If the models are linear, returns the mean and sd weights of highest weight 10 features
#     3. If the models are not linear, returns the permutation importance results for:
#         - Correlated and non-correlated OTUs:
#         - Top 10 features or feature groups will be listed
#         - Mean percent AUROC change from original AUROC after permutation
get_interp_info <- function(data, model_name){
  if("key" %in% colnames(data)){
    # If the models are linear, we saved the weights of every OTU for each datasplit
    # We want to plot the ranking of OTUs for linear models.
    # 1. Get dataframe transformed into long form
    #         The OTU names are in 1 column(repeated for 100 datasplits)
    #         The weight value are in 1 column(for each of the datasplits)
    imp_first_20 <- data %>%
      # 2. Group by the OTU name and compute mean and sd for each OTU
      group_by(key) %>%
      summarise(median_rank = median(rank)) %>%
      arrange(median_rank) %>%
      head(n=5) %>%
      select(key, median_rank)

    imp <- data %>%
      filter(key %in% imp_first_20$key) %>%
      group_by(key)

  }
  # If we want to calculate the permutation importance results
  # Then we use the files without the weight information but the permutation results
  else{
    if("names" %in% colnames(data)){ # If the file has non-correlated OTUs
      correlated_data <- data %>%
        # 1. Group by the OTU names and calculate median and sd for auc change
        group_by(names) %>%
        summarise(imp = median(new_auc), sd_imp = sd(new_auc))
      # 4.  a) Order the dataframe from largest weights to smallest.
      #     b) Select the largest 10
      #     c) Put the signs back to weights
      #     d) select the OTU names, median weights with their signs and the sd
      imp <- correlated_data %>%
        arrange(imp)
    }
    else if("X1" %in% colnames(data)){
      # The file doesn't have "names" column which means these are correlated OTU groups
      # The file has correlated OTUs and their total percent auc change per group in one row
      # Each row has different groups of OTUs that are correlated together
      #     1. We will group by the first OTU (since it is only present in one group only)
      #         This will group all the datasplits for that OTU group together
      #     2. We then get the mean percent auc change of that correlated OTU group
      correlated_data <- data %>%
        group_by(X1) %>%
        summarise(imp = median(new_auc), sd_imp = sd(new_auc))
      #     3. We will now only take the first 10 and add the other OTUs to the row.
      #       We have the mean percent auc change for each correlated group of OTUs in a row
      #       We will also have all the OTU names in the group in the same row.
      imp <- correlated_data %>%
        arrange(imp) %>%
        inner_join(data, by="X1") %>% # order the largest 10
        unique() %>%
        select(-new_auc, -model)
    }
      else{
        print("linear model")
        imp <- NULL
        }

  }
  return(imp)
}
# -------------------------------------------------------------------->


# Summarise walltime
summarise_walltime <- function(files){
  summarized_walltimes <- summarise(files, mean_walltime = mean(files[,1]), sd_AUC = sd(files[,1]))
  return(summarized_walltimes)
}

# Calculate unpaired two-samples Wilcoxon test to see if models differ from one another signigicantly
wilcoxon_test <- function(data, model_name_1, model_name_2){
  wilcox_result <- wilcox.test((data %>% filter(model==model_name_1))$test_aucs, (data %>% filter(model==model_name_2))$test_aucs)
  return(wilcox_result)
}
