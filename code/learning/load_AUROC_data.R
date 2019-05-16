######################################################################
#--------- Load .tsv data to get mean test AUC for 7 models--------#
######################################################################

# Read in the cvAUCs, testAUCs for 100 splits.
best_files <- list.files(path= '../data/process', pattern='combined_best.*', full.names = TRUE)
all <- map_df(best_files, read_files) 

# Get the The unpaired two-samples Wilcoxon test to see if models differ from one another signigicantly

rf_xgboost <- wilcoxon_test(all, "Random_Forest", "XGBoost")
rf_logit <- wilcoxon_test(all, "Random_Forest", "L2_Logistic_Regression")
rf_L2svm <- wilcoxon_test(all, "Random_Forest", "L2_Linear_SVM")
xgboost_L2svm <- wilcoxon_test(all, "XGBoost", "L2_Linear_SVM")
logit_L2svm <- wilcoxon_test(all, "L2_Logistic_Regression", "L2_Linear_SVM")

# Bind them together and summarise mean testAUC by model
test_all <- all %>%
  melt_data() %>% 
  select(-variable) %>% 
  group_by(model, Performance) %>% 
  summarise(mean_AUC = mean(AUC), sd_AUC = sd(AUC)) %>% 
  filter(Performance=="testing") 

# Get the order index from small to large meanAUC
performance_index <- order(test_all$mean_AUC)

# Bind them together and summarise mean cvAUC by model
cv_all <- all %>%
  melt_data() %>% 
  select(-variable) %>% 
  group_by(model, Performance) %>% 
  summarise(mean_AUC = mean(AUC), sd_AUC = sd(AUC)) %>% 
  filter(Performance=="cross-validation") 

# Get the difference between mean cvAUC and testAUC for each model
difference <- cv_all$mean_AUC - test_all$mean_AUC[match(cv_all$model, test_all$model)]
difference_model <- data.frame(difference, test_all$model)
# Get the order index from small to large of the differences
difference_index <- order(abs(difference_model$difference))


