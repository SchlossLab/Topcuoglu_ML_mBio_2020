# Author: Begum Topcuoglu
# Date: 2018-06-06
#
######################################################################
# This script plots the feature rankings
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



get_feature_ranked_files <- function(file_name, model_name){
  importance_data <- read_tsv(file_name)
  ranks <- get_interp_info(importance_data, model_name) %>%
    as.data.frame()
  return(ranks)
}

plot_feature_ranks <- function(data){
  
  otus <- data %>% 
    group_by(key) %>% 
    summarise(imp = median(rank)) %>% 
    arrange(imp) 
  
  otus <- otus$key %>% 
        as.character()
  
  # Most changed OTU at the top, followed by others ordered by median
  data$key <- factor(data$key,levels = c(otus[5], otus[4], otus[3], otus[2], otus[1]))
  
  plot <- ggplot(data, aes(data$key, data$rank)) +
    geom_point(color = 'orange1') +
    stat_summary(fun.y = "median", colour = 'orangered4', geom = "point", size = 2.5) +
    coord_flip() +
    scale_y_continuous(limits=c(0, 100)) +
    theme_classic() +
    theme(plot.margin=unit(c(1.5,3,1.5,3),"mm"),
          legend.position="none",
          axis.title = element_text(size=10),
          axis.text = element_text(size=10),
          panel.border = element_rect(colour = "black", fill=NA, size=1),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.text.x=element_text(size = 8, colour='black'),
          axis.text.y=element_text(size = 8, colour='black'),
          axis.title.x=element_blank())
  return(plot)
}

######################################################################
#--------------Run the functions and plot feature ranks ----------#
######################################################################


L1_SVM_imp <- get_feature_ranked_files("data/process/combined_L1_Linear_SVM_feature_ranking.tsv", "L1_Linear_SVM")
l1_svm_graph <- plot_feature_ranks(L1_SVM_imp)  +
  scale_x_discrete(name = "L1 Linear SVM")


L2_SVM_imp <- get_feature_ranked_files("data/process/combined_L2_Linear_SVM_feature_ranking.tsv", "L2_Linear_SVM")
l2_svm_graph <- plot_feature_ranks(L2_SVM_imp)+
  scale_x_discrete(name = "L2 Linear SVM ")

logit_imp <- get_feature_ranked_files("data/process/combined_L2_Logistic_Regression_feature_ranking.tsv", "L2_Logistic_Regression")
logit_graph <- plot_feature_ranks(logit_imp) +
  scale_x_discrete(name = "L2 Logistic 
Regression ")
# -------------------------------------------------------------------->


######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################
#combine with cowplot

linear <- plot_grid(logit_graph, l1_svm_graph, l2_svm_graph, labels = c("A", "B", "C"), align = 'v', ncol = 1)

ggdraw(add_sub(linear, "Feature Ranks", vpadding=grid::unit(0,"lines"), size=10, x=0.6))

ggsave("Figure_3.png", plot = last_plot(), device = 'png', path = 'submission', width = 3, height = 5)
