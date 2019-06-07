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
    head(5) %>%
    mutate(rank = 1:nrow(.)) %>% 
    mutate(model=name)
    
  return(data_first_10)
}

x <- list("Decision_Tree", "Random_Forest", "XGBoost")

total <- map_df(x, grab_importance) %>% group_by(model)

# Grouped
ggplot(total, aes(y=rank, x=names, fill=model)) +
  geom_bar(stat="identity", width=0.35, position = position_dodge2(preserve = "single")) +
  geom_point(position = position_dodge(width=0.25), size=1, aes(color=model)) +
  scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9", "black")) +
  scale_color_manual(values=c("#999999", "#E69F00", "#56B4E9", "black")) +
  coord_flip() +
  scale_x_discrete(name="Top 5 OTUs") +
  scale_y_continuous(name="ranked importance based on 
  permutation importance effect on AUROC") +
  theme_classic() 

ggsave("Figure_S3.png", plot = last_plot(), device = 'png', path = 'submission', width = 5, height = 2)  

