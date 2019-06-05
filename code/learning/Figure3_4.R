# Author: Begum Topcuoglu
# Date: 2018-02-13
#
######################################################################
# This script looks at the model interpretation
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
#----------------- Define the functions we will use -----------------#
######################################################################


# -------------------- Read files ------------------------------------>
# This function:
#     1. takes a list of files(with their path)
#     2. reads them as delim files with comma seperator
#     3. returns the dataframe
read_files <- function(filenames){
  for(file in filenames){
    # Read the files generated in main.R 
    # These files have cvAUCs and testAUCs for 100 data-splits
    data <- read.delim(file, header=T, sep=',')
  }
  return(data)
}
# -------------------------------------------------------------------->


# ------------------- Re-organize feature importance  ----------------->
# This function:
#     1. Takes in a dataframe (different data for each model) and the model name
#     2. If the models are linear, returns the mean and sd weights of highest weight 10 features
#     3. If the models are not linear, returns the permutation importance results for:
#         - Correlated and non-correlated OTUs:
#         - Top 10 features or feature groups will be listed
#         - Mean percent AUROC change from original AUROC after permutation
get_interp_info <- function(data, model_name){ 
  if("key" %in% colnames(data)){ 
    # If the models are linear, we saved the weights of every OTU for each datasplit
    # We want to plot the ranking of OTUs for linear models. 
    # 1. Get dataframe transformed into long form
    #         The OTU names are in 1 column(repeated for 100 datasplits)
    #         The weight value are in 1 column(for each of the datasplits)
    imp <- data %>% 
      # 2. Group by the OTU name and compute mean and sd for each OTU
      group_by(key) %>% 
      summarise(mean_rank = median(rank)) %>% 
      arrange(mean_rank) %>% 
      head(n=20) %>% 
      select(key, mean_rank)
  }
  # If we want to calculate the permutation importance results
  # Then we use the files without the weight information but the permutation results
  else{ 
    if("names" %in% colnames(data)){ # If the file has non-correlated OTUs 
      correlated_data <- data %>% 
        # 1. Group by the OTU names and calculate median and sd for auc change 
        group_by(names) %>% 
        summarise(imp = median(new_auc), sd_imp = sd(new_auc)) 
      # 4.  a) Order the dataframe from largest weights to smallest.
      #     b) Select the largest 10 
      #     c) Put the signs back to weights
      #     d) select the OTU names, median weights with their signs and the sd
      imp <- correlated_data %>% 
        arrange(imp)
    }
    else if("X1" %in% colnames(data)){
      # The file doesn't have "names" column which means these are correlated OTU groups
      # The file has correlated OTUs and their total percent auc change per group in one row
      # Each row has different groups of OTUs that are correlated together
      #     1. We will group by the first OTU (since it is only present in one group only)
      #         This will group all the datasplits for that OTU group together
      #     2. We then get the mean percent auc change of that correlated OTU group
      correlated_data <- data %>% 
        group_by(X1) %>% 
        summarise(imp = median(new_auc), sd_imp = sd(new_auc))
      #     3. We will now only take the first 10 and add the other OTUs to the row.
      #       We have the mean percent auc change for each correlated group of OTUs in a row
      #       We will also have all the OTU names in the group in the same row.
      imp <- correlated_data %>% 
        arrange(imp) %>% 
        inner_join(data, by="X1") %>% # order the largest 10 
        unique() %>% 
        select(-new_auc, -model)
    }
      else{
        print("linear model")
        imp <- NULL
        }
    
  }
  return(imp)
}
# -------------------------------------------------------------------->



######################################################################
#--------------Run the functions and plot importance ----------#
######################################################################

# ----------- Read in saved combined feature importances ---------->
# List the important features files by defining a pattern in the path
# Correlated files are:
cor_files <- list.files(path= 'data/process', pattern='combined_all_imp_features_cor_.*', full.names = TRUE)
# Non-correlated files are:
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

get_feature_ranked_files <- function(file_name, model_name){
  importance_data <- read_tsv(file_name)
  get_interp_info(importance_data, model_name) %>% 
    as.data.frame() %>% 
    write_tsv(., paste0("data/process/", model_name, "_rank_importance.tsv"))
}

L1_SVM_imp <- get_feature_ranked_files("data/process/combined_L1_Linear_SVM_feature_ranking.tsv", "L1_Linear_SVM") %>% 
  arrange(mean_rank) %>% 
  mutate(rank = 1:nrow(.)) %>% 
  select(key, rank) %>% 
  head(10)

L2_SVM_imp <- get_feature_ranked_files("data/process/combined_L2_Linear_SVM_feature_ranking.tsv", "L2_Linear_SVM") %>% 
  arrange(mean_rank) %>% 
  mutate(rank = 1:nrow(.)) %>% 
  select(key, rank) %>% 
  head(10)

logit_imp <- get_feature_ranked_files("data/process/combined_L2_Logistic_Regression_feature_ranking.tsv", "L2_Logistic_Regression") %>% 
  arrange(mean_rank) %>% 
  mutate(rank = 1:nrow(.)) %>% 
  select(key, rank) %>% 
  head(10)
# -------------------------------------------------------------------->

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
  data_first_ten <- read.delim(paste0("data/process/", name, "_non_cor_importance.tsv"), header=T, sep='\t') %>%
    arrange(imp) %>% 
    head(10)
  # Get the new test aucs for 100 datasplits for each OTU permuted
  data_full <- read_files(paste0("data/process/combined_all_imp_features_non_cor_results_", name, ".csv")) %>%
    # Only keep the OTUs and their AUCs for the ones that are in the top 5 changed (decreased the most) ones
    filter(names %in% data_first_ten$names) %>% 
    group_by(names)
    # Keep the 5 OTU names as a list to use in the plot as factors
  otus <- data_first_ten$names %>% 
    droplevels() %>% 
    as.character()
  # Base auc at the top, then followed by the most changed OTU, followed by others ordered by median
  data_full$names <- factor(data_full$names,levels = c(otus[10], otus[9], otus[8], otus[7], otus[6], otus[5], otus[4], otus[3], otus[2], otus[1]))
  # Plot boxplot
  lowerq <-  quantile(data_base$new_auc)[2]
  upperq <-  quantile(data_base$new_auc)[4]
  median <-  median(data_base$new_auc) %>% 
    data.frame()
  

  plot <- ggplot() +
    geom_boxplot(data=data_full, aes(x=names, y=new_auc), fill="#E69F00",  alpha=0.8) +
    geom_rect(aes(ymin=lowerq, ymax=upperq, xmin=0, xmax=Inf), fill="grey") +
    geom_boxplot(data=data_full, aes(x=names, y=new_auc), fill="#E69F00", alpha=0.8) +
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
          axis.title = element_text(size=14),
          axis.text = element_text(size=12),
          panel.border = element_rect(colour = "black", fill=NA, size=1), 
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.text.x=element_text(size = 12, colour='black'),
          axis.text.y=element_text(size = 10, colour='black'), 
          axis.title.x=element_blank()) 
  
  #-----------------------Save median info ------------------------ #
  #-----------------------Save top 5 features ------------------------ #
  write_tsv(data_full, paste0("data/process/", name, "_non_linear_top_ten_importance.tsv"))
  write_tsv(median, paste0("data/process/", name, "_non_linear_base_median.tsv"))
  
  # Check if correlated OTUs make a difference in AUROC

  #data_cor_results <- read.delim(paste0("data/process/", name, "_cor_importance.tsv"), header=T, sep='\t') %>%
   # filter(!imp==data_base_means$imp) 
  #if(nrow(data_cor_results)==0) {
  #  print("Correlation dataframe empty. No need to plot correlated OTUs. Plot only non-correlated OTUs.")
  #}else{
   # print("Investigate correlated OTUs effect and plot both.")
  #}
  return(plot)
}
# ----------------------------------------------------------------------->


logit_plot <- base_nonlin_plot(logit, "L2_Logistic_Regression") +
  scale_x_discrete(name = "L2 Logistic Regression ") 

l1_plot <- base_nonlin_plot(l1svm, "L1_Linear_SVM") +
  scale_x_discrete(name = "L1 Linear SVM") 

l2_plot <- base_nonlin_plot(l2svm, "L2_Linear_SVM") +
  scale_x_discrete(name = "L2 Linear SVM ") 

# ----------------------------------------------------------------------->

# ----------------- SVM with radial basis function------------------------>
# Plot most important 5 features effect on AUROC
rbf_plot <- base_nonlin_plot(rbf, "RBF_SVM") +
  scale_x_discrete(name = "RBF SVM ") 
# ----------------------------------------------------------------------->

# --------------------------- Decision Tree ----------------------------->
# Plot most important 5 features effect on AUROC
dt_plot <- base_nonlin_plot(dt, "Decision_Tree") +
  scale_x_discrete(name = "Decision tree ") 
# ----------------------------------------------------------------------->

# --------------------------- Random Forest ----------------------------->
# Plot most important 5 features effect on AUROC
rf_plot <- base_nonlin_plot(rf, "Random_Forest") +
  scale_x_discrete(name = "Random forest ") 
# ----------------------------------------------------------------------->

# --------------------------- XGBoost ----------------------------->
# Plot most important 5 features effect on AUROC
xgboost_plot <- base_nonlin_plot(xgboost, "XGBoost")+
  scale_x_discrete(name = "XGBoost ") 
# ----------------------------------------------------------------------->

######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################
#combine with cowplot
linear <- plot_grid(l1svm_plot, l2svm_plot, logit_plot, labels = c("A", "B", "C"), align = 'h', ncol = 3)

ggsave("Figure_3.png", plot = linear, device = 'png', path = 'submission', width = 11, height = 2.5)

non_lin <- plot_grid(rbf_plot, dt_plot, rf_plot, xgboost_plot, labels = c("A", "B", "C", "D"))
ggdraw(add_sub(non_lin, "AUROC with the OTU permuted randomly", vpadding=grid::unit(0,"lines"), y=5, x=0.5, vjust=4.5))

ggsave("Figure_4.png", plot = last_plot(), device = 'png', path = 'submission', width = 8, height = 6)





