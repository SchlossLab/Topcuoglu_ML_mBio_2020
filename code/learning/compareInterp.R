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

get_linear_interp_info <- function(linear_model){ 
  ranked_linear_model <- linear_model %>% 
  select(-normal) %>% 
  group_by(names) %>% 
  summarise(mean_imp = mean(cancer), sd_imp = sd(cancer), n = n()) %>% 
  arrange(-n) %>% 
  head(n=10)
  return(ranked_linear_model)
}

for(files in interp_files){
  print(files)
  file <- read_files(files)
  model_name <- as.character(file$model[1])
  dataframe <- get_linear_interp_info(file) 
  write_tsv(dataframe, paste0("data/process/", model_name, "_importance.tsv"))
}
    


