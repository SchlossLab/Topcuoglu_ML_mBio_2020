deps = c("pROC","randomForest","AUCRF","knitr","rmarkdown","vegan","gtools", "tidyverse");
for (dep in deps){
  if (dep %in% installed.packages()[,"Package"] == FALSE){
    install.packages(as.character(dep), quiet=TRUE);
  }
  library(dep, verbose=FALSE, character.only=TRUE)
}
meta <- read.delim('data/metadata.tsv', header=T, sep='\t') %>% 
  select(sample, dx)
  
shared <- read.delim('data/baxter.0.03.subsample.shared', header=T, sep='\t') %>% 
   select(-label, -numOtus)

data <- inner_join(meta, shared, by=c("sample"="Group")) %>% 
  filter(dx != 'adenoma') %>% 
  mutate(dx = ifelse(dx == "normal",0,1)) %>% 
  select(-sample)
  

require(caTools)
set.seed(101) 
sample = sample.split(data, SplitRatio = .75)
train = subset(data, sample == TRUE)
test  = subset(data, sample == FALSE)
