# Author: Begum Topcuoglu
# Date: 2018-06-05
#
######################################################################
# This script plots permutation importances of 7 models
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
library(ggrepel)
######################################################################
#----------------- Call the functions we will use -----------------#
######################################################################

source("code/learning/functions.R")

######################################################################
#--------------Run the functions and plot importance ----------#
######################################################################

grab_importance <-  function(name){
  
  # Grab the names of the OTUs that have the lowest median AUC when they are permuted
  data_all <- read.delim(paste0("data/process/", name, "_non_cor_importance.tsv"), header=T, sep='\t') %>%
    arrange(imp) %>% 
    head(50) %>% 
    mutate(model=name) %>% 
    mutate(new_auc = imp) %>%
    select(-sd_imp, -imp) 
  return(data_all)
}

grab_base_auc <-  function(name){
    # Grab the base test auc values for 100 datasplits
  data_base <- read.delim(paste0("data/process/combined_best_hp_results_", name, ".csv"), header=T, sep=',') %>%
    select(-cv_aucs) %>% 
    group_by(model) %>% 
    summarise(median_base=median(test_aucs))
  return(data_base)
}

get_delta_auc <- function(name){
  new_total <- total %>% 
    filter(model==name) %>% 
    mutate(delta= (base_aucs %>% filter(model==name))$median_base - (total %>% filter(model == name))$new_auc)
  return(new_total)
}
  

x <- list("L1_Linear_SVM", "L2_Linear_SVM", "L2_Logistic_Regression", "RBF_SVM", "Decision_Tree", "Random_Forest", "XGBoost")

# Read in the cvAUCs, testAUCs for 100 splits as base test_aucs
base_aucs <- map_df(x, grab_base_auc)

total <- map_df(x, grab_importance)

new_total <- map_df(x, get_delta_auc) %>% 
  select(-new_auc) %>% 
  group_by(model)



  # Grouped
ggplot(new_total, aes(y=delta, x=model, fill = model)) +
  geom_dotplot(binaxis = 'y', stackdir = "center", stackratio=1.5, dotsize=0.3, binpositions = "all" ) +
  scale_fill_brewer(palette="Dark2") +
  theme_bw() +
  geom_label_repel(aes(label = names),
                   data = subset(new_total, delta>0.013),
                   box.padding   = 0.35, 
                   point.padding = 0.5,
                   direction     = "x",
                   segment.color = 'grey50') +
  guides(fill = guide_legend(override.aes = aes(label = ""))) +
  scale_y_continuous(name = expression(Delta~"AUROC after permutation"),
                     limits=c(0, 0.1)) +
  scale_x_discrete(name = "",
                   labels=c("Decision tree",
                            "L1 linear SVM",
                            "L2 linear SVM",
                            "L2 logistic regression",
                            "Random Forest",
                            "RBF SVM",
                            "XGBoost")) +
  theme(plot.margin=unit(c(1.1,1.1,1.1,1.1),"cm"),
        legend.justification=c(1,0),
        legend.position=c(1,0.4),
        #legend.position="bottom",
        legend.title = element_blank(),
        legend.background = element_rect(linetype="solid", color="black", size=0.5),
        legend.box.margin=margin(c(12,12,12, 12)),
        legend.text=element_text(size=12),
        #legend.title=element_text(size=22),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        text = element_text(size = 12),
        axis.text.x=element_text(size = 15, colour='black'),
        axis.text.y=element_text(size = 20, colour='black'),
        axis.title.y=element_text(size = 24),
        axis.title.x=element_blank(),
        panel.border = element_rect(linetype="solid", colour = "black", fill=NA, size=1.5))


ggsave("Figure_S3.png", plot = last_plot(), device = 'png', path = 'submission', width = 15, height = 5)  

