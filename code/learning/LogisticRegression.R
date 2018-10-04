deps = c("caret","knitr","rmarkdown","vegan","gtools", "tidyverse");
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
  select(-sample) 
  
data$dx <- factor(data$dx, labels=c("normal", "cancer"))


inTraining <- createDataPartition(data$dx, p = .80, list = FALSE)
training <- data[ inTraining,] 
testing  <- data[-inTraining,]
x_train <- training %>% select(-dx)
y_train <- ifelse(training$dx == "cancer", 1, 0)
y_train <- as.factor(y_train)
fitControl <- trainControl(method = "repeatedcv",number = 5,repeats = 1)

set.seed(825)
model <- train(x=x_train, y=y_train,method = "glm", trControl = fitControl, verbose=TRUE, warnings())

  
set.seed(20081989) 
sample = sample.split(data$dx, SplitRatio = .80)
train = subset(data, sample == TRUE)
test  = subset(data, sample == FALSE)



x_train <- model.matrix(dx~., train)[,-1]
y_train <- ifelse(train$dx == "cancer", 1, 0)
x_test <- model.matrix(dx~., test)[,-1]
y_test <- ifelse(test$dx == "cancer", 1, 0)
## Hyper-parameter tuning
# prepare training scheme
# prepare training scheme
cv.lasso <- cv.glmnet(x_train, y_train, nfolds=5, alpha = 1, family = "binomial", type.measure = "auc")
model <- glmnet(x_train, y_train, alpha = 1, family = "binomial",
                lambda = cv.lasso$lambda.min)




