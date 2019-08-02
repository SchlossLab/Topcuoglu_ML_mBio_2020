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
# ------------------- Re-organize feature importance  ----------------->
# This function:
#     1. Takes in a dataframe (different data for each model) and the model name
#     2. If the models are linear, returns the median rank of the top ranked 5 features
#     3. If the models are not linear, returns the permutation importance results for:
#         - Correlated and non-correlated OTUs:
#         - Top 5 features or feature groups will be listed
#         - New AUROC which will differ from original AUROC after permutation
get_score_info <- function(data, model_name){
    # If the models are linear, we used get_feature_rankings.R and then mege_feature_ranks.sh first
    # The created file after those 2 steps will be used in this function,
    # Data format is:
    #         The OTU names are in 1 column(repeated for 100 datasplits)
    #         The ranks based on absolute weights are in 1 column(for each of the datasplits)
    #		  The weight value is on another column
    # We want to use/plot only the top 5 highest ranked OTUs
    # Initial step is to get which are the highest 5 ranked OTUs by looking at their median rank
    # 1. We group by OTU name to make sure we are taking all the data-splits into account
    signed_data <- data %>%
      # 2. Group by the OTU name and compute median rank for each OTU
      group_by(key) %>%
      summarise(mean_score  = mean(value), sd_score = sd(value)) %>% 
      mutate(sign = case_when(mean_score<0 ~ "negative",
                              mean_score>0 ~ "positive",
                              mean_score==0 ~ "zero"))
      
      signed_data$mean_score <- abs(signed_data$mean_score)
      
      scores <- signed_data %>%
        arrange(desc(mean_score)) %>%
        mutate(mean_score = case_when(sign=="negative" ~ mean_score*-1,
                                 sign=="positive"~ mean_score,
                                 sign=="zero" ~ mean_score)) %>%
        select(key, mean_score, sd_score)
      
      top_scores <- scores %>% 
        head(n=10) 
    
  return(top_scores)    
}


get_feature_score_files <- function(file_name, model_name){
  importance_data <- read_tsv(file_name)
  scores <- get_score_info(importance_data, model_name) %>%
    as.data.frame()
  return(scores)
}

# This function:
#     1. Top 10 highest z-scores for 100 splits)
#     2. Returns a plot. Each datapoint is the rank of the OTU at one datasplit.

plot_feature_scores <- function(data){
  plot <- ggplot(data, aes(fct_reorder(data$key, abs(data$mean_score)), data$mean_score)) +
    geom_point(colour = "brown2", size = 3) +
    theme_classic() +
    coord_flip() +
    theme(legend.text=element_text(size=18),
          legend.title=element_text(size=22),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          text = element_text(size = 12),
          axis.text.x=element_text(size = 12, colour='black'),
          axis.text.y=element_text(size = 12, colour='black'),
          axis.title.y=element_text(size = 13),
          axis.title.x=element_text(size = 13), 
          panel.border = element_rect(colour = "black", fill=NA, size=1)) +
    geom_hline(yintercept=0, linetype="dashed", 
               color = "black")
  return(plot)
}


# This function:
#     1. Grabs the taxonomy information for all OTUs
#     2. Grabs the top 5 important taxa and their feature rankings
#     3. Merges the 2 tables (taxa and importance) by the OTU name
#     4. Make a new column that merges taxa info and OTU name
#     5. Turns that to a list to use as labels in the plot

get_taxa_info_as_labels <- function(data){
  

  # Names of the y-axis labels
  taxa_info <- read.delim('data/baxter.taxonomy', header=T, sep='\t') %>% 
    select(-Size) %>% 
    mutate(key=OTU) %>% 
    select(-OTU)
  
  taxa_otus <- inner_join(data, taxa_info, by="key") %>% 
    mutate(taxa=gsub("(.*);.*","\\1",Taxonomy)) %>% 
    mutate(taxa=gsub("(.*)_.*","\\1",Taxonomy)) %>% 
    mutate(taxa=gsub("(.*);.*","\\1",Taxonomy)) %>% 
    mutate(taxa=gsub(".*;","",taxa)) %>% 
    mutate(taxa=gsub("(.*)_.*","\\1",taxa)) %>% 
    mutate(taxa=str_remove_all(taxa, "[(100)]")) %>% 
    select(key, taxa, mean_score) %>% 
    unite(key, taxa, key, sep="(") %>% 
    mutate(key = paste(key,")", sep=""))
  
  
  taxa_otus <- taxa_otus$key %>% 
    as.character()
  
  return(taxa_otus)
}

######################################################################
#--------------Run the functions and plot feature scores ----------#
######################################################################


L2_SVM_imp <- get_feature_score_files("data/process/combined_L2_Linear_SVM_feature_scores.tsv", "L2_Linear_SVM")

l2_svm_graph <- plot_feature_scores(L2_SVM_imp)  +
  scale_x_discrete(name = expression(paste(L[2], "-regularized linear SVM")),
                   labels = get_taxa_info_as_labels(L2_SVM_imp)) +
  scale_y_continuous(name="Feature weight z-scores",
                     limits = c(-10, 10),
                     breaks = seq(-10, 10, 2)) +
  geom_errorbar(aes(ymin=L2_SVM_imp$mean_score-L2_SVM_imp$sd_score, 
                    ymax=L2_SVM_imp$mean_score+L2_SVM_imp$sd_score),
                width = 0.1)




######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################
#combine with cowplot

linear <- plot_grid(l1_svm_graph, l2_svm_graph, logit_graph, labels = c("A", "B", "C"), align = 'v', ncol = 1)

ggdraw(add_sub(linear, "Feature Ranks", vpadding=grid::unit(0,"lines"), y=5, x=0.7, vjust=4.75, size=15))

ggsave("zscore.png", plot = last_plot(), device = 'png', path = 'submission', width = 9, height = 6)
