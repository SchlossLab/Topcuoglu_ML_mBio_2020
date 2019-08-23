# Author: Begum Topcuoglu
# Date: 2018-02-13
#
######################################################################
# This script plots permutation importance results for non-linear models
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
# ----------------------------------------------------------------------------------->

# ----------- Loops to re-organize feature importance info -------------------------->
# This loop will:
#   1. Read the model files saved in 'interp_files' list
#   2. Get the model name from the file
#   3. Use te get_interp_info() for each model. 
#   4. Save the top 5 features and their median, sd importance value in a .tsv file

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
# ----------------------------------------------------------------------------->


# ----------- Get the actual testing AUROC values before permutation ---------->

# Read in the cvAUCs, testAUCs for 100 splits as base test_aucs
best_files <- list.files(path= 'data/process', pattern='combined_best.*', full.names = TRUE)

logit <- read_files(best_files[4])
l2svm <- read_files(best_files[3])
l1svm <- read_files(best_files[2])
rbf <- read_files(best_files[6])
rf <- read_files(best_files[5])
dt <- read_files(best_files[1])
xgboost <- read_files(best_files[7])
# ----------------------------------------------------------------------------->


# --------  Get the top 5 OTUs that have the largest impact on AUROC ---------->

# Define the function to get the  most important top 5 OTUs
top_20_otus <-  function(name){
  
  # Order the dataframe from smallest new_auc to largest.
  # Because the smallest new_auc means that that OTU decreased AUC a lot when permuted
  
  data_first_20 <- read.delim(paste0("data/process/", name, "_non_cor_importance.tsv"),
                                header=T, 
                                sep='\t') %>%
    arrange(imp) %>% 
    head(20)
  
  return(data_first_20)
}

# --------------------------- RBF_SVM ------------------------------>
rbf_top <- top_20_otus("RBF_SVM")
# --------------------------- Decision Tree ------------------------>
dt_top <- top_20_otus("Decision_Tree")
# --------------------------- Random Forest ------------------------>
rf_top <- top_20_otus("Random_Forest")
# --------------------------- XGBoost ------------------------------>
xgboost_top <- top_20_otus("XGBoost")
# ----------------------------------------------------------------------------->


# --------  Find the overlapping important OTUs between non-linear models ----->
intersect_four <- Reduce(intersect, list(dt_top$names, rf_top$names, xgboost_top$names, rbf_top$names))
intersect_three <- setdiff(Reduce(intersect, list(dt_top$names, rf_top$names, xgboost_top$names)), intersect_four)


intersect_rf_dt <- setdiff(intersect(dt_top$names,rf_top$names), c(intersect_four, intersect_three))

intersect_xgboost_dt <- setdiff(intersect(dt_top$names,xgboost_top$names), c(intersect_four, intersect_three))

intersect_xgboost_rf <- setdiff(intersect(rf_top$names,xgboost_top$names), c(intersect_four, intersect_three))

intersect_rbf_dt <- setdiff(intersect(rbf_top$names,dt_top$names), c(intersect_four, intersect_three,intersect_xgboost_dt))

intersect_rbf_xgboost <- setdiff(intersect(xgboost_top$names,rbf_top$names), c(intersect_four, intersect_three,intersect_xgboost_dt))

# ----------------------------------------------------------------------------->


######################################################################
#-------------- Plot the permutation importance of all models ----------#
######################################################################
# -----------------------Base plot function -------------------------->
# Define the base plot 
    # data: the base testing AUROC dataframe
    # name: name of the model
    # list: overlapping OTUs list
base_nonlin_plot <-  function(data, name){
  
  # Grab the base test auc values for 100 datasplits
  data_base <- data %>% 
    select(-cv_aucs) %>% 
    mutate(new_auc = test_aucs) %>% 
    mutate(names="base_auc") %>% 
    select(-test_aucs)
  
  # Have a median base auc value for h-line and for correlated testing
  data_base_medians <- data %>% 
    summarise(imp = median(test_aucs), sd_imp = sd(test_aucs)) %>% 
    mutate(names="base_auc")
  
  
  # Grab the names of the OTUs that have the 5 lowest median AUC when they are permuted
  # These will be the 5 OTUs that have effect the testing AUROC the most when permuted
  data_first_20 <- read.delim(paste0("data/process/", name, "_non_cor_importance.tsv"), header=T, sep='\t') %>%
    arrange(imp) %>% 
    head(20)
  
  # Get the new test aucs for 100 datasplits for each OTU permuted
  # So the full dataset with importance info on non-correlated OTUs
  data_full <- read_files(paste0("data/process/combined_all_imp_features_non_cor_results_", name, ".csv")) %>%
    # Keep OTUs and their AUCs for the ones that are in the top 5 changed 
    # (decreased the most) 
    filter(names %in% data_first_20$names) %>% 
    # Create a seperate column to show which OTUs are overlapping
    mutate(common_otus = case_when(names %in% intersect_four ~ "four",
                                   names %in% intersect_three ~ "three",
                                   names %in% intersect_rf_dt ~ "rf_dt",
                                   names %in% intersect_xgboost_dt ~ "xgboost_dt",
                                   names %in% intersect_xgboost_rf ~ "xgboost_rf",
                                   names %in% intersect_rbf_xgboost ~ "rbf_xgboost",
                                   TRUE ~ "diff")) %>% 
    group_by(names)
  
  # Plot boxplot for the base test_auc values
  lowerq <-  quantile(data_base$new_auc)[2]
  upperq <-  quantile(data_base$new_auc)[4]
  median <-  median(data_base$new_auc) %>% 
    data.frame()
  
  # Define colors for overlapping OTUs
  cols <- c("four" = "orange", "three" = "lightsalmon", "rf_dt" = "lightblue", "xgboost_dt" = "darkseagreen3", "xgboost_rf"="red", "rbf_xgboost" = "hotpink" , "diff"="white")

  # Plot the figure
  plot <- ggplot() +
    geom_boxplot(data=data_full, aes(fct_reorder(names, -new_auc), new_auc, fill=factor(common_otus)), alpha=0.7) +
    geom_rect(aes(ymin=lowerq, ymax=upperq, xmin=0, xmax=Inf), fill="grey") +
    geom_boxplot(data=data_full, aes(x=names, y=new_auc, fill=factor(common_otus)), alpha=0.7) +
    scale_fill_manual(values=cols) +
    geom_hline(yintercept = data_base_medians$imp , linetype="dashed") +
    #geom_hline(yintercept = upperq, alpha=0.5) +
    #geom_hline(yintercept = lowerq, alpha=0.5) +
    coord_flip() +
    theme_classic() +
    scale_y_continuous(name = " AUROC with the OTU permuted randomly", 
                       limits = c(0.5,1), 
                       expand=c(0,0)) +
    theme(plot.margin=unit(c(1.5,3,1.5,3),"mm"),
          legend.position="none",
          axis.title = element_text(size=20),
          axis.text = element_text(size=20),
          panel.border = element_rect(colour = "black", fill=NA, size=2), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.text.y=element_text(size = 12, colour='black'), 
          axis.title.x=element_blank(),
          axis.text.x=element_blank(), 
          axis.ticks = element_line(colour = "black", size = 1.1)) 
  
  #----------------------- Take a look at correlated OTUs ------------------------ #
  # Check if the top 5 correlated OTU groups make a difference in AUROC
  if(name == "L1_Linear_SVM" || name =="L2_Linear_SVM" || name =="L2_Logistic_Regression"){
    print("For linear model, we did NOT save the results of correlated OTUs permutation")
  }
  else{
    data_cor_results <- read.delim(paste0("data/process/", name, "_cor_importance.tsv"), header=T, sep='\t') %>%
    filter(imp!=data_base_medians$imp) 
        if(nrow(data_cor_results)==0) {
          print("The testing AUROC does not change when correlated groups are permuted. No need to plot correlated OTUs. Plot only non-correlated OTUs.")
        } else{
    print("Investigate correlated OTUs effect and plot both.")
        }
     }
  return(plot)
}
# ------------------------------------------------------------------------>

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
    filter(names %in% data_first_20$names) %>% 
    arrange(-imp)
  
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
    mutate(taxa=gsub('(_).*', '', taxa)) %>% 
    select(names, taxa, imp) %>% 
    unite(names, taxa, names, sep="(") %>% 
    arrange(desc(imp)) %>% 
    mutate(names = paste(names, ")", sep="")) %>% 
    mutate(names=paste0(gsub('TU0*', 'TU ', names))) 
  
  return(taxa_otus$names)
}

# ----------------- SVM with radial basis function------------------------>
# Plot most important 5 features effect on AUROC
rbf_plot <- base_nonlin_plot(rbf, "RBF_SVM") +
  scale_x_discrete(name = "RBF SVM ",
                   labels = get_taxa_info_as_labels("RBF_SVM")) 
# ----------------------------------------------------------------------->

# --------------------------- Decision Tree ----------------------------->
# Plot most important 5 features effect on AUROC
dt_plot <- base_nonlin_plot(dt, "Decision_Tree") +
  scale_x_discrete(name = "Decision tree ",
                   labels = get_taxa_info_as_labels("Decision_Tree")) 
# ----------------------------------------------------------------------->

# --------------------------- Random Forest ----------------------------->
# Plot most important 5 features effect on AUROC
rf_plot <- base_nonlin_plot(rf, "Random_Forest") +
  scale_x_discrete(name = "Random forest ",
                   labels = get_taxa_info_as_labels("Random_Forest")) 
# ----------------------------------------------------------------------->

# --------------------------- XGBoost ----------------------------->
# Plot most important 5 features effect on AUROC
xgboost_plot <- base_nonlin_plot(xgboost, "XGBoost")+
  scale_x_discrete(name = "XGBoost ",
                   labels = get_taxa_info_as_labels("XGBoost")) +
  theme(axis.text.x=element_text(size = 16, colour='black'))
# ----------------------------------------------------------------------->

######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################
#combine with cowplot
perm_tree_based <- plot_grid(rbf_plot, dt_plot, rf_plot, xgboost_plot, labels = c("A", "B", "C", "D"), cols=2, scale = 0.97, align = "v")
ggdraw(add_sub(perm_tree_based, "AUROC with the OTU permuted randomly", size=18, vpadding=grid::unit(0,"lines"), y=5, x=0.6, vjust=4.75))

ggsave("Figure_4.png", plot = last_plot(), device = 'png', path = 'submission', width = 12, height = 8)



