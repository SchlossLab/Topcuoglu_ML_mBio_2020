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

melt_data <-  function(data) {
  data_melt <- data %>%
    melt(measure.vars=c('cv_aucs', 'test_aucs')) %>%
    rename(AUC=value) %>%
    mutate(Performance = case_when(variable == "cv_aucs" ~ 'cross-validation', variable == "test_aucs" ~ 'testing')) %>%
    group_by(Performance)
  return(data_melt)
}

# Read in files as delim that are saved in a list with a pattern
read_files <- function(filenames){
  for(file in filenames){
    # Read the files generated in main.R
    # These files have cvAUCs and testAUCs for 100 data-splits
    data <- read.delim(file, header=T, sep=',')
  }
  return(data)
}


# Get model name with sub from file name
get_model_name <- function(files){
  pat1 <- "data/process/walltime_"
  name_files <- sub(pat1, "", files)
  pat2 <- ".csv"
  names <- sub(pat2, "", name_files)
  return(names)
}

summarise_walltime <- function(files){
  summarized_walltimes <- summarise(files, mean_walltime = mean(files[,1]), sd_AUC = sd(files[,1])) 
  return(summarized_walltimes)
}
