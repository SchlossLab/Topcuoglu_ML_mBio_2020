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
  if(model_name=="L2_Logistic_Regression" || 
     model_name=="L1_Linear_SVM" || 
     model_name=="L2_Linear_SVM"){
    # If the models are linear, we saved the weights of every OTU for each datasplit
    # 1. Get dataframe transformed into long form
    #         The OTU names are in 1 column(repeated for 100 datasplits)
    #         The weight value are in 1 column(for each of the datasplits)
    weights <- data %>% 
      select(-Bias, -model) %>% 
      gather(factor_key=TRUE) %>% 
    # 2. Group by the OTU name and compute mean and sd for each OTU
      group_by(key) %>% 
      summarise(mean_weights = mean(value), sd_weights = sd(value)) %>% 
      # 2. We now want to save to a new column the sign of the weights
      mutate(sign = case_when(mean_weights<0 ~ "negative",
                              mean_weights>0 ~ "positive",
                              mean_weights==0 ~ "zero")) 
    # 3. We change all the weights to their absolute value
    #       Because we want to see which weights are the largest 
    weights$mean_weights <- abs(weights$mean_weights)
    # 4.  a) Order the dataframe from largest weights to smallest.
    #     b) Select the largest 10 
    #     c) Put the signs back to weights
    #     d) select the OTU names, mean weights with their signs and the sd
    imp_means <- weights %>% 
      arrange(desc(mean_weights)) %>% 
      head(n=10) %>% 
      mutate(mean_weights = case_when(sign=="negative" ~ mean_weights*-1,
                                      sign=="positive"~ mean_weights)) %>% 
      select(key, mean_weights, sd_weights)
    
  }
  # If models are not linear then we will read in permutation importance results
  else{ 
    if("names" %in% colnames(data)){ # If the file has non-correlated OTUs 
      correlated_data <- data %>% 
        # 1. Group by the OTU names and calculate mean and sd for auc change 
        group_by(names) %>% 
        summarise(mean_imp = mean(new_auc), sd_imp = sd(new_auc)) %>% 
        # 2. We now want to save to a new column the sign of the weights
        mutate(sign = case_when(mean_imp<0 ~ "negative",
                                mean_imp>0 ~ "positive",
                                mean_imp==0 ~ "zero")) 
      # 3. We change all the percent changes to their absolute value
      #       Because we want to see which are the largest weigths
      correlated_data$mean_imp <- abs(correlated_data$mean_imp)
      # 4.  a) Order the dataframe from largest weights to smallest.
      #     b) Select the largest 10 
      #     c) Put the signs back to weights
      #     d) select the OTU names, mean weights with their signs and the sd
      imp_means <- correlated_data %>% 
        arrange(desc(mean_imp)) %>% 
        head(n=10) %>% 
        mutate(mean_imp = case_when(sign=="negative" ~ mean_imp*-1,
                                    sign=="positive"~ mean_imp)) %>% 
        select(-sign)
    }
    else{
      # The file doesn't have "names" column which means these are correlated OTU groups
      # The file has correlated OTUs and their total percent auc change per group in one row
      # Each row has different groups of OTUs that are correlated together
      #     1. We will group by the first OTU (since it is only present in one group only)
      #         This will group all the datasplits for that OTU group together
      #     2. We then get the mean percent auc change of that correlated OTU group
      correlated_data <- data %>% 
        group_by(X1) %>% 
        summarise(mean_imp = mean(new_auc), sd_imp = sd(new_auc))
      #     3. We will now only take the first 10 and add the other OTUs to the row.
      #       We have the mean percent auc change for each correlated group of OTUs in a row
      #       We will also have all the OTU names in the group in the same row.
      imp_means <- correlated_data %>% 
        arrange(-mean_imp) %>% 
        head(n=10) %>% 
        inner_join(data, by="X1") %>% # order the largest 10 
        unique() %>% 
        select(-new_auc, -model)
    }
  }
  return(imp_means)
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
  model_name <- as.character(importance_data$model[1]) # get the model name from table
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
# -------------------------------------------------------------------->



######################################################################
#-------------- Plot the weights of linear models ----------#
######################################################################

# -----------------------Base plot function -------------------------->
# We will plot the mean feature weights for top 10 OTUs
# Define the base plot for the linear modeling methods
base_plot <-  function(data, x_axis, y_axis){
  plot <- ggplot(data, aes(fct_reorder(x_axis, -abs(y_axis)), y_axis)) +
    geom_point(colour = "brown2", size = 3) +
    theme_classic() +
    scale_x_discrete(name = "") +
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
# ----------------------------------------------------------------------->


# ------------------L1 SVM with linear kernel---------------------------->
l1svm <- read.delim("data/process/L1_Linear_SVM_non_cor_importance.tsv", header=T, sep='\t') 

l1svm_plot <- base_plot(l1svm, x=l1svm$key,y=l1svm$mean_weights) +
  scale_y_continuous(name="L1 linear kernel SVM feature weights",
                    limits = c(-3, 3),
                    breaks = seq(-3, 3, 0.5)) +
  geom_errorbar(aes(ymin=l1svm$mean_weights-l1svm$sd_weights, 
                    ymax=l1svm$mean_weights+l1svm$sd_weights), 
                width=.01) 
# ----------------------------------------------------------------------->

# ------------------L2 SVM with linear kernel---------------------------->
l2svm <- read.delim("data/process/L2_Linear_SVM_non_cor_importance.tsv", header=T, sep='\t') 
l2svm_plot <- base_plot(l2svm, x=l2svm$key,y=l2svm$mean_weights) +
  scale_y_continuous(name="L2 linear kernel SVM feature weights",
                     limits = c(-1, 1),
                     breaks = seq(-1, 1, 0.5)) +    
  geom_errorbar(aes(ymin=l2svm$mean_weights-l2svm$sd_weights, 
                    ymax=l2svm$mean_weights+l2svm$sd_weights), 
                width=.01) 
# ----------------------------------------------------------------------->


# ------------------- L2 logistic regression ---------------------------->
logit <- read.delim("data/process/L2_Logistic_Regression_non_cor_importance.tsv", header=T, sep='\t') 
logit_plot <- base_plot(logit, x=logit$key, y=logit$mean_weights) +
  scale_y_continuous(name="L2 logistic regression coefficients",
                     limits = c(-4, 4),
                     breaks = seq(-4, 4, 0.5)) +   
  geom_errorbar(aes(ymin=logit$mean_weights-logit$sd_weights, 
                    ymax=logit$mean_weights+logit$sd_weights), 
                width=.01)
# ----------------------------------------------------------------------->


######################################################################
#-------------- Plot the importance of non-linear models ----------#
######################################################################


# ----------------- SVM with radial basis function------------------------>
# Plot correlated OTUs importance rbf svm
rbf_cor_results <- read.delim("data/process/RBF_SVM_cor_importance.tsv", header=T, sep='\t') %>% 
  filter(!mean_imp==0) 

rbf_plot <- read.delim("data/process/RBF_SVM_non_cor_importance.tsv", header=T, sep='\t') %>% 
  ggplot(aes(x=reorder(names, mean_imp), y=mean_imp, label=mean_imp)) +
  geom_bar(stat='identity')+
  coord_flip() +
  theme_classic() +
  scale_y_continuous(name = "Normalized mean feature importance ") +
  scale_x_discrete(name = "Random forest ") +
  theme(legend.position="none",
        axis.title = element_text(size=14),
        axis.text = element_text(size=12),
        panel.border = element_rect(colour = "black", fill=NA, size=1), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text.x=element_text(size = 12, colour='black'),
        axis.text.y=element_text(size = 10, colour='black'))
# ----------------------------------------------------------------------->

# Plot decision tree
#dt <- read.delim("data/process/Decision_Tree_importance.tsv", header=T, sep='\t') %>% 
#base_plot_nonlin(names, mean_imp)
dt_cor_results <- read.delim("data/process/Decision_Tree_cor_importance.tsv", header=T, sep='\t') %>% 
  filter(!mean_imp==0) 
  
dt_plot <- read.delim("data/process/Decision_Tree_non_cor_importance.tsv", header=T, sep='\t') %>% 
  head(n=5) %>% 
  ggplot(aes(x=reorder(names, mean_imp), y=mean_imp, label=mean_imp)) +
  geom_bar(stat='identity')+
  coord_flip() +
  theme_classic() +
  scale_y_continuous(name = "Percent AUROC decrease", 
                     limits = c(0, 100)) +
  scale_x_discrete(name = "Decision tree ") +
  theme(legend.position="none",
        axis.title = element_text(size=10),
        axis.text = element_text(size=10),
        panel.border = element_rect(colour = "black", fill=NA, size=1), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text.x=element_text(size = 10, colour='black'),
        axis.text.y=element_text(size = 10, colour='black'))
# ----------------------------------------------------------------------->

# Plot random forest

rf_cor_results <- read.delim("data/process/Random_Forest_cor_importance.tsv", header=T, sep='\t') %>% 
  filter(!mean_imp==0) 

rf_plot <- read.delim("data/process/Random_Forest_non_cor_importance.tsv", header=T, sep='\t') %>% 
  ggplot(aes(x=reorder(names, mean_imp), y=mean_imp, label=mean_imp)) +
  geom_bar(stat='identity')+
  coord_flip() +
  theme_classic() +
  scale_y_continuous(name = "Normalized mean feature importance ") +
  scale_x_discrete(name = "Random forest ") +
  theme(legend.position="none",
        axis.title = element_text(size=14),
        axis.text = element_text(size=12),
        panel.border = element_rect(colour = "black", fill=NA, size=1), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text.x=element_text(size = 12, colour='black'),
        axis.text.y=element_text(size = 10, colour='black'))

  

# Plot xgboost
xgboost_cor_results <- read.delim("data/process/XGBoost_cor_importance.tsv", header=T, sep='\t') %>% 
  filter(!mean_imp==0) 

xgboost_plot <- read.delim("data/process/XGBoost_non_cor_importance.tsv", header=T, sep='\t') %>% 
  ggplot(aes(x=reorder(names, mean_imp), y=mean_imp, label=mean_imp)) +
  geom_bar(stat='identity')+
  coord_flip() +
  theme_classic() +
  scale_y_continuous(name = "Normalized mean feature importance ") +
  scale_x_discrete(name = "Random forest ") +
  theme(legend.position="none",
        axis.title = element_text(size=14),
        axis.text = element_text(size=12),
        panel.border = element_rect(colour = "black", fill=NA, size=1), 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.text.x=element_text(size = 12, colour='black'),
        axis.text.y=element_text(size = 10, colour='black'))




######################################################################
#-----------------------Save figure as .pdf ------------------------ #
######################################################################
#combine with cowplot
linear <- plot_grid(logit_plot, l1svm_plot, l2svm_plot, labels = c("A", "B", "C"))

ggsave("Figure_3a.pdf", plot = linear, device = 'pdf', path = 'results/figures', width = 18, height = 10)

non_lin <- plot_grid(dt_plot, labels = c("A"))

ggsave("Figure_3b.pdf", plot = non_lin, device = 'pdf', path = 'results/figures', width = 5, height = 2)





