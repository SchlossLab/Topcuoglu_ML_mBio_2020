deps = c("pROC","randomForest","AUCRF","knitr","rmarkdown","vegan","gtools");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}
meta <- read.delim('data/metadata.tsv', header=T, sep='\t')
shared <- read.delim('data/baxter.0.03.subsample.shared', header=T, sep='\t')

all_data <- merge(meta, shared, by.x='sample', by.y='Group')
all_data$lesion <- factor(NA, levels=c(0,1))
all_data$lesion[all_data$dx=='cancer'] <- 1
all_data$lesion[all_data$dx=='normal'] <- 0

pval <- function(p){
  if(p < 0.001){p <- 'p<0.001'}
  else{p <- sprintf('p=%.1g', p)}
  return(p)
}

otu_canc_data <- all_data[meta$dx!='adenoma',c('lesion',colnames(all_data)[grep('Otu[0123456789]', colnames(all_data))])]

set.seed(62515)
otu_canc_model <- AUCRF(lesion ~ ., data=otu_canc_data, ranking='MDA', ntree=500, pdel=0.05)
canc_opt_model <- otu_canc_model$RFopt
otu_canc_probs <- predict(canc_opt_model, type='prob')
otu_canc_roc <- roc(otu_canc_data$lesion~otu_canc_probs[,2])

canc_fit_roc <- roc(all_data[all_data$dx!='adenoma','lesion'] ~ all_data[all_data$dx!='adenoma','fit_result'])
