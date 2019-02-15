
qsub L1_Linear_SVM.pbs
while [ ! -f data/temp/all_hp_results_L1_Linear_SVM_98.csv ]; do sleep 1; done
