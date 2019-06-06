# Author: Begum Topcuoglu
# Date: 2018-06-05
#
######################################################################
# This script plots permutation importances as ranks and fingerprints over 7 models
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
#----------------- Call the functions we will use -----------------#
######################################################################

source("code/learning/functions.R")

######################################################################
#--------------Run the functions and plot importance ----------#
######################################################################

grab_importance <-  function(name){
  # Grab the names of the OTUs that have the lowest median AUC when they are permuted
  data_first_10 <- read.delim(paste0("data/process/", name, "_non_cor_importance.tsv"), header=T, sep='\t') %>%
    arrange(imp) %>% 
    head(10) %>%
    mutate(rank = 1:nrow(.)) %>% 
    mutate(model=name)
    
  return(data_first_10)
}

rbf <- grab_importance("RBF_SVM")
dt <- grab_importance("Decision_Tree")
rf <- grab_importance("Random_Forest")
xgboost <-  grab_importance("XGBoost")

total <- bind_rows(dt, rf, xgboost) %>% 
  group_by(model)


# Grouped
ggplot(total, aes(y=rank, x=names, color=model)) + 
  geom_point(position=position_dodge(width = 0.90), size=3) +
  geom_segment(aes(xend=names), yend=0, position=position_dodge(width = 0.90), size=1) +
  expand_limits(y=0) +
  scale_color_manual(values=c("#999999", "#E69F00", "#56B4E9")) +
  coord_flip() 

