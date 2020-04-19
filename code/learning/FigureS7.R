# Author: Begum Topcuoglu
# Date: 2018-02-13
#
######################################################################
# This script plots permutation importance results of linear models
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

# ----------- Read in saved combined feature importances ---------->
# List the important features files by defining a pattern in the path

# Correlated files are:
# For linear models these files have the feature weights and coefficients.
# For non-linear models these files have the correlated OTUs grouped together
cor_files <- list.files(path= 'data/process', pattern='combined_all_imp_features_cor_.*', full.names = TRUE)

# Non-correlated files are:
# For linear models these files have the non-correlated OTUs permutation importance results
# For non-linear models these files have non-correlated OTUs permutation importance results
non_cor_files <-  list.files(path= 'data/process', pattern='combined_all_imp_features_non_cor_.*', full.names = TRUE)
# -------------------------------------------------------------------->

# ----------- Loops to re-organize feature importance info ---------->
# This loop will:
#   1. Read the model files saved in 'interp_files' list
#   2. Get the model name from the file
#   3. Use te get_interp_info() for each model. 
#   4. Save the top 10 features and their mean, sd importance value in a .tsv file

for(file_name in cor_files){
  importance_data <- read_files(file_name)
  model_name <- as.character(importance_data$model[1])# get the model name from table
  get_interp_info(importance_data, model_name) %>% 
    as.data.frame() %>% 
    write_tsv(., paste0("data/process/", model_name, "_cor_importance.tsv"))
}

for(file_name in non_cor_files){
  importance_data <- read_files(file_name)
  model_name <- as.character(importance_data$model[1]) # get the model name from table
  get_interp_info(importance_data, model_name) %>% 
    as.data.frame() %>% 
    write_tsv(., paste0("data/process/", model_name, "_non_cor_importance.tsv"))
}



# Read in the cvAUCs, testAUCs for 100 splits as base test_aucs
best_files <- list.files(path= 'data/process', pattern='combined_best.*', full.names = TRUE)


logit <- read_files(best_files[4])
l2svm <- read_files(best_files[3])
l1svm <- read_files(best_files[2])
rbf <- read_files(best_files[6])
rf <- read_files(best_files[5])
dt <- read_files(best_files[1])
xgboost <- read_files(best_files[7])


######################################################################
#-------------- Plot the permutation importance of all models ----------#
######################################################################
# -----------------------Base plot function -------------------------->
# Define the base plot 
base_nonlin_plot <-  function(data, name){
  # Grab the base test auc values for 100 datasplits
  data_base <- data %>% 
    select(-cv_aucs) %>% 
    mutate(new_auc = test_aucs) %>% 
    mutate(names="base_auc") %>% 
    select(-test_aucs)
  # Have a median base auc value for h-line and for correlated testing
  data_base_means <- data %>% 
    summarise(imp = median(test_aucs), sd_imp = sd(test_aucs)) %>% 
    mutate(names="base_auc")
  # Grab the names of the OTUs that have the lowest median AUC when they are permuted
  data_first_20 <- read.delim(paste0("data/process/", name, "_non_cor_importance.tsv"), header=T, sep='\t') %>%
    arrange(imp) %>% 
    head(20)
  # Get the new test aucs for 100 datasplits for each OTU permuted
  data_full <- read_files(paste0("data/process/combined_all_imp_features_non_cor_results_", name, ".csv")) %>%
    # Only keep the OTUs and their AUCs for the ones that are in the top 5 changed (decreased the most) ones
    filter(names %in% data_first_20$names) %>% 
    group_by(names)
    # Keep the 5 OTU names as a list to use in the plot as factors

  # Plot boxplot
  lowerq <-  quantile(data_base$new_auc)[2]
  upperq <-  quantile(data_base$new_auc)[4]
  median <-  median(data_base$new_auc) %>% 
    data.frame()
  

  plot <- ggplot() +
    geom_boxplot(data=data_full, aes(fct_reorder(names, -new_auc), y=new_auc), fill="white",  alpha=0.8) +
    geom_rect(aes(ymin=lowerq, ymax=upperq, xmin=0, xmax=Inf), fill="grey") +
    geom_boxplot(data=data_full, aes(x=names, y=new_auc), fill="white", alpha=0.8) +
    geom_hline(yintercept = data_base_means$imp , linetype="dashed") +
    #geom_hline(yintercept = upperq, alpha=0.5) +
    #geom_hline(yintercept = lowerq, alpha=0.5) +
    coord_flip() +
    theme_classic() +
    scale_y_continuous(name = " AUROC with the OTU permuted randomly", 
                       limits = c(0.5,1), 
                       expand=c(0,0)) +
    theme(plot.margin=unit(c(1.5,3,1.5,3),"mm"),
          legend.position="none",
          axis.title = element_text(size=10),
          axis.text = element_text(size=8),
          panel.border = element_rect(colour = "black", fill=NA, size=2), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.text.y=element_text(size = 8, colour='black'), 
          axis.title.x=element_blank(),
          axis.text.x=element_blank(), 
          axis.ticks = element_line(colour = "black")) 

  return(plot)
}
# ----------------------------------------------------------------------->

# -----------------------Taxonomy Info Function -------------------------->
# data: the full dataset with importance info on non-correlated OTUs
#       It has new test aucs for 100 datasplits for each OTU permuted
get_taxa_info_as_labels <- function(name){
  
  
  # Get the new test aucs for 100 datasplits for each OTU permuted
  # So the full dataset with importance info on non-correlated OTUs
  data <- read_files(paste0("data/process/combined_all_imp_features_non_cor_results_", name, ".csv"))
  
  # Grab the names of the OTUs that have the 5 lowest median AUC when they are permuted
  # These will be the 5 OTUs that have effect the testing AUROC the most when permuted
  data_first_20 <- read.delim(paste0("data/process/", name, "_non_cor_importance.tsv"), header=T, sep='\t') %>%
    arrange(imp) %>% 
    head(20)
  
  # Grab the names of the top 5 OTUs in the order of their median rank  
  otus <- data %>% 
    group_by(names) %>% 
    summarise(imp = median(new_auc)) %>% 
    arrange(imp) %>% 
    filter(names %in% data_first_20$names) 
  
  # Names of the y-axis labels
  taxa_info <- read.delim('data/baxter.taxonomy', header=T, sep='\t') %>% 
    select(-Size) %>% 
    mutate(names=OTU) %>% 
    select(-OTU)
  
  taxa_otus <- inner_join(otus, taxa_info, by="names") %>% 
    mutate_if(is.character, str_to_upper) %>% 
    mutate(taxa=gsub("(.*);.*","\\1",Taxonomy)) %>% 
    mutate(taxa=gsub("(.*)_.*","\\1",Taxonomy)) %>% 
    mutate(taxa=gsub("(.*);.*","\\1",Taxonomy)) %>% 
    mutate(taxa=gsub(".*;","",taxa)) %>% 
    mutate(taxa=gsub("(.*)_.*","\\1",taxa)) %>% 
    mutate(taxa=str_remove_all(taxa, "[(100)]")) %>% 
    mutate(taxa=gsub('[0-9]+', '', taxa)) %>% 
    select(names, taxa, imp) %>% 
    unite(names, taxa, names, sep=" (") %>% 
    arrange(desc(imp)) %>% 
    mutate(names = paste(names, ")", sep="")) %>% 
    mutate(names=paste0(gsub('TU0*', 'TU ', names))) 
  
  
  taxa_otus_names <- taxa_otus$names %>% 
    as.character()
  
  return(taxa_otus_names)
}
# ----------------------------------------------------------------------->

# --------------------- Linear models ----------------------------------->
logit_plot <- base_nonlin_plot(logit, "L2_Logistic_Regression") +
  scale_x_discrete(name = expression(paste(L[2], "-regularized logistic regression")),
                   labels = get_taxa_info_as_labels("L2_Logistic_Regression")) +
  theme(axis.text.x=element_text(size = 8, colour='black'))

l1_plot <- base_nonlin_plot(l1svm, "L1_Linear_SVM") +
  scale_x_discrete(name = expression(paste(L[1], "-regularized linear SVM")),
                   labels = get_taxa_info_as_labels("L1_Linear_SVM")) 

l2_plot <- base_nonlin_plot(l2svm, "L2_Linear_SVM") +
  scale_x_discrete(name = expression(paste(L[2], "-regularized linear SVM")),
                   labels = get_taxa_info_as_labels("L2_Linear_SVM"))  
# ----------------------------------------------------------------------->

######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################
#combine with cowplot
linear <- plot_grid(l1_plot, l2_plot, logit_plot, labels = c("A", "B", "C"), align = 'v', ncol = 1, scale = 0.97)

ggdraw(add_sub(linear, "AUROC with the OTU permuted randomly", size=12, vpadding=grid::unit(0,"lines"), y=5, x=0.65, vjust=4.75))

ggsave("Figure_S7.tiff", plot = last_plot(), device = 'tiff', path = 'submission', width = 6, height = 9, dpi=300)




