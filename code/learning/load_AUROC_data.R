
######################################################################
#--------- Load cv and test AUROCs of 7 models for 100 datasplits--------#
######################################################################

# Read in the cvAUCs, testAUCs for 100 splits.
best_files <- list.files(path= '../data/process', pattern='combined_best.*', full.names = TRUE)
all <- map_df(best_files, read_files) 

# Get the The unpaired two-samples Wilcoxon test to see if test_aucs of models differ from one another signigicantly
# Load the wilcoxon test function from ../code/learning/functions.R
rf_xgboost <- wilcoxon_test(all, "Random_Forest", "XGBoost")
rf_logit <- wilcoxon_test(all, "Random_Forest", "L2_Logistic_Regression")
rf_L2svm <- wilcoxon_test(all, "Random_Forest", "L2_Linear_SVM")
rf_rbf <- wilcoxon_test(all, "Random_Forest", "RBF_SVM")
xgboost_L2svm <- wilcoxon_test(all, "XGBoost", "L2_Linear_SVM")
logit_L2svm <- wilcoxon_test(all, "L2_Logistic_Regression", "L2_Linear_SVM")
rbf_L2svm <- wilcoxon_test(all, "RBF_SVM", "L2_Linear_SVM")
l1svm_L2svm <- wilcoxon_test(all, "L1_Linear_SVM", "L2_Linear_SVM")
dt_L2svm <- wilcoxon_test(all, "Decision_Tree", "L2_Linear_SVM")

# Bind them together and summarise mean testAUC by model
test_all <- all %>%
  melt_data() %>% 
  select(-variable) %>% 
  group_by(model, Performance) %>% 
  summarise(median_AUC = median(AUC), iqr_AUC = IQR(AUC)) %>% 
  filter(Performance=="testing") 

# Get the order index from small to large medianAUC
performance_index <- order(test_all$median_AUC)

# Bind them together and summarise median cvAUC by model
cv_all <- all %>%
  melt_data() %>% 
  select(-variable) %>% 
  group_by(model, Performance) %>% 
  summarise(median_AUC = median(AUC), iqr_AUC = IQR(AUC)) %>% 
  filter(Performance=="cross-validation") 

# Get the difference between mean cvAUC and testAUC for each model
difference <- cv_all$median_AUC - test_all$median_AUC[match(cv_all$model, test_all$model)]
difference_model <- data.frame(difference, test_all$model)
# Get the order index from small to large of the differences
difference_index <- order(abs(difference_model$difference))


# Compute what happens when OTU367 is permuted in random forest and decision tree

# These will be the 5 OTUs that have effect the testing AUROC the most when permuted
DT <- read.delim(paste0("../data/process/Decision_Tree_non_cor_importance.tsv"), header=T, sep='\t') %>%
  arrange(imp) %>% 
  filter(names=="Otu00367") 

RF <- read.delim(paste0("../data/process/Random_Forest_non_cor_importance.tsv"), header=T, sep='\t') %>%
  arrange(imp) %>% 
  filter(names=="Otu00367") 

