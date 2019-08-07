# Author: Begum Topcuoglu
# Date: 2018-06-06
#
######################################################################
# This script plots the feature rankings for linear models
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
#----------------- Define the functions we will use -----------------#
######################################################################

# ------------------- Re-organize feature importance  ----------------->
# This function:
#     1. Takes in a combined (100 split) feature rankings for each model) and the model name
#     2. Returns the top 5 ranked (1-5 with 1 being the highest ranked) OTUs (ranks of the OTU for 100 splits)
get_feature_ranked_files <- function(file_name, model_name){
  importance_data <- read_tsv(file_name)
  ranks <- get_interp_info(importance_data, model_name) %>%
    as.data.frame()
  return(ranks)
}

# This function:
#     1. Top 5 ranked (1-5 lowest rank) OTUs (ranks of the OTU for 100 splits)
#     2. Returns a plot. Each datapoint is the rank of the OTU at one datasplit.
                        
plot_feature_ranks <- function(data){
    # Plot from highest median ranked OTU to least (only top 5) and thir ranks that lay between 1-100
    # Rank 1 is the highest rank
    plot <- ggplot(data, aes(reorder(data$key, -data$rank, FUN = median), data$rank)) +
      geom_point(aes(colour= factor(data$sign)), size=2) + # datapoints lighter color
      scale_color_manual(values=c("red3", "#56B4E9", "#999999")) +
      stat_summary(fun.y = function(x) median(x), colour = 'black', geom = "point", size = 3) + # Median darker
      coord_flip() +
      scale_y_continuous(limits=c(0, 100)) + # Only keep the first 100 ranks (truncated)
      theme_classic() +
      theme(plot.margin=unit(c(1.5,3,1.5,3),"mm"),
          legend.position="none",
          panel.border = element_rect(colour = "black", fill=NA, size=1),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.title.y=element_text(size = 11, colour='black', face="bold"), 
          axis.text.y=element_text(size = 8, colour='black'))
    return(plot)
}

# This function:
#     1. Grabs the taxonomy information for all OTUs
#     2. Grabs the top 5 important taxa and their feature rankings
#     3. Merges the 2 tables (taxa and importance) by the OTU name
#     4. Make a new column that merges taxa info and OTU name
#     5. Turns that to a list to use as labels in the plot

get_taxa_info_as_labels <- function(data){
  
  # Grab the names of the top 5 OTUs in the order of their median rank  
  otus <- data %>% 
    group_by(key) %>% 
    summarise(imp = median(rank)) %>% 
    arrange(-imp) 
  # Names of the y-axis labels
  taxa_info <- read.delim('data/baxter.taxonomy', header=T, sep='\t') %>% 
   select(-Size) %>% 
    mutate(key=OTU) %>% 
    select(-OTU)

  taxa_otus <- inner_join(otus, taxa_info, by="key") %>% 
    mutate(taxa=gsub("(.*);.*","\\1",Taxonomy)) %>% 
    mutate(taxa=gsub("(.*)_.*","\\1",Taxonomy)) %>% 
    mutate(taxa=gsub("(.*);.*","\\1",Taxonomy)) %>% 
    mutate(taxa=gsub(".*;","",taxa)) %>% 
    mutate(taxa=gsub("(.*)_.*","\\1",taxa)) %>% 
    mutate(taxa=str_remove_all(taxa, "[(100)]")) %>% 
    select(key, taxa, imp) %>% 
    unite(key, taxa, key, sep="(") %>% 
    mutate(key = paste(key,")", sep=""))
    
  return(taxa_otus$key)
}

######################################################################
#--------------Run the functions and plot feature ranks ----------#
######################################################################


L1_SVM_imp <- get_feature_ranked_files("data/process/combined_L1_Linear_SVM_feature_ranking.tsv", "L1_Linear_SVM")
l1_svm_graph <- plot_feature_ranks(L1_SVM_imp)  +
  scale_x_discrete(name = expression(paste(L[1], "-regularized linear SVM")),
                   labels = get_taxa_info_as_labels(L1_SVM_imp))


L2_SVM_imp <- get_feature_ranked_files("data/process/combined_L2_Linear_SVM_feature_ranking.tsv", "L2_Linear_SVM")
l2_svm_graph <- plot_feature_ranks(L2_SVM_imp)+
  scale_x_discrete(name = expression(paste(L[2], "-regularized linear SVM")),
                   labels = get_taxa_info_as_labels(L2_SVM_imp))

logit_imp <- get_feature_ranked_files("data/process/combined_L2_Logistic_Regression_feature_ranking.tsv", "L2_Logistic_Regression")
logit_graph <- plot_feature_ranks(logit_imp) +
  scale_x_discrete(name = expression(paste(L[2], "-regularized logistic regression")),
                   labels = get_taxa_info_as_labels(logit_imp)) +
  theme(axis.text.x=element_text(size = 12, colour='black'))
# -------------------------------------------------------------------->


######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################
#combine with cowplot

linear <- plot_grid(l1_svm_graph, l2_svm_graph, logit_graph, labels = c("A", "B", "C"), align = 'v', ncol = 1)

ggdraw(add_sub(linear, "Feature Ranks", vpadding=grid::unit(0,"lines"), y=5, x=0.7, vjust=4.75, size=15))

ggsave("Figure_3.png", plot = last_plot(), device = 'png', path = 'submission', width = 6, height = 9.2)
