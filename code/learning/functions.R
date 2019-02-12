# Author: Begum Topcuoglu
# Date: 2018-12-06
#
######################################################################
# Place to store useful functions that will be used repeatedly throughout
######################################################################

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


