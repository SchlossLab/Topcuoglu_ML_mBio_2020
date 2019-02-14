#!/bin/bash
SEARCH_DIR=data/temp
FINAL_DIR=data/process
# Keep the first line of File1 and remove the first line of all the others and combine

for model in "XGBoost" "Random_Forest" "RBF_SVM"  "L1_Linear_SVM" "L2_Logistic_Regression" "L2_Linear_SVM" "Decision_Tree"
do
  	head -1 $SEARCH_DIR/all_hp_results_"$model"_1.csv  > $SEARCH_DIR/combined_all_hp_results_"$model".csv; tail -n +2 $

        head -1 $SEARCH_DIR/best_hp_results_"$model"_1.csv  > $SEARCH_DIR/combined_best_hp_results_"$model".csv; tail -n +$

        head -1 $SEARCH_DIR/all_imp_features_results_"$model"_1.csv > $SEARCH_DIR/combined_all_imp_features_results_"$mode$
        mv $SEARCH_DIR/combined_all_hp_results_"$model".csv $FINAL_DIR/combined_all_hp_results_"$model".csv
        mv $SEARCH_DIR/combined_best_hp_results_"$model".csv $FINAL_DIR/combined_best_hp_results_"$model".csv
        mv $SEARCH_DIR/combined_all_imp_features_results_"$model".csv $FINAL_DIR/combined_all_imp_features_results_"$model$
done

#rm $SEARCH_DIR/all_hp_*
#rm $SEARCH_DIR/best_hp_*
