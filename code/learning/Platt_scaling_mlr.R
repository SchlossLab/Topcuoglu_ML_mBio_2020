devtools::install_github("mlr-org/mlr")
library(tidyverse)
library(mlr)
library(stringi)
######################## DATA PREPARATION #############################
# Features: Hemoglobin levels and 16S rRNA gene sequences in the stool 
# Labels: - Colorectal lesions of 490 patients. 
#         - Defined as cancer or not.(Cancer here means: SRN)
# Read in metadata and select only sample Id and diagnosis columns
meta <- read.delim('data/metadata.tsv', header=T, sep='\t') %>%
  select(sample, Dx_Bin, fit_result)
# Read in OTU table and remove label and numOtus columns
shared <- read.delim('data/baxter.0.03.subsample.shared', header=T, sep='\t') %>%
  select(-label, -numOtus)
# Merge metadata and OTU table.
# Group advanced adenomas and cancers together as cancer and normal, high risk normal and non-advanced adenomas as normal
# Then remove the sample ID column
data <- inner_join(meta, shared, by=c("sample"="Group")) %>%
  mutate(dx = case_when(
    Dx_Bin== "Adenoma" ~ "normal",
    Dx_Bin== "Normal" ~ "normal",
    Dx_Bin== "High Risk Normal" ~ "normal",
    Dx_Bin== "adv Adenoma" ~ "cancer",
    Dx_Bin== "Cancer" ~ "cancer"
  )) %>%
  select(-sample, -Dx_Bin) %>%
  drop_na()
# We want the diagnosis column to a factor
data$dx <- factor(data$dx, labels=c("normal", "cancer"))
###################################################################


inTraining <- caret::createDataPartition(data$dx, p = .80, list = FALSE)
training <- data[ inTraining,]
testing  <- data[-inTraining,]
# Scale all features between 0-1
preProcValues <- caret::preProcess(training, method = "range")
trainTransformed <- predict(preProcValues, training)
testTransformed <- predict(preProcValues, testing)

testTransformed$dx <- as.factor(ifelse(testTransformed$dx == "cancer", 1, 0))
trainTransformed$dx <- as.factor(ifelse(trainTransformed$dx == "cancer", 1, 0))

#create a task
trainTask <- mlr::makeClassifTask(data = trainTransformed,target = "dx")
testTask <- mlr::makeClassifTask(data = testTransformed,target = "dx")
#load svm
getParamSet("classif.LiblineaRL1L2SVC") #do install kernlab package 
l1svm <- makeLearner("classif.LiblineaRL1L2SVC", predict.type = "response")
#Set parameters
pssvm <- makeParamSet(
  makeDiscreteParam("cost", values = 2^c(-8,-4,-2,0))) #cost parameters

#specify search function
ctrl <- makeTuneControlGrid()
set_cv <- makeResampleDesc("CV",iters = 3L)
#tune model
res <- tuneParams(l1svm, task = trainTask, resampling = set_cv, par.set = pssvm, control = ctrl,measures = acc)
#set the model with best params
t.svm <- setHyperPars(l1svm, par.vals = res$x)
#train
par.svm <- train(l1svm, trainTask)
#test
predict.svm <- predict(par.svm, testTask)

lrn=makeClassificationViaRegressionWrapper(l1svm, predict.type = "prob")
mod <- mlr::train(lrn, trainTask)

makeClassificationViaRegressionWrapper = function(learner, predict.type = "response") {
  learner = checkLearner(learner, "classif")
  lrn = makeBaseWrapper(
    id = stri_paste(learner$id, "classify", sep = "."),
    type = "classif",
    next.learner = learner,
    package = "mlr",
    par.set = makeParamSet(),
    par.vals = list(),
    learner.subclass = "ClassificationViaRegressionWrapper",
    model.subclass = "ClassificationViaRegressionModel"
  )
  lrn$predict.type = predict.type
  return(lrn)
}
trainLearner.ClassificationViaRegressionWrapper = function(.learner, .task, .subset = NULL, .weights = NULL, ...) {
  pos = getTaskDesc(.task)$positive
  td = getTaskData(.task, target.extra = TRUE, subset = .subset)
  target.name = stri_paste(pos, "prob", sep = ".")
  data = td$data
  data[[target.name]] = ifelse(td$target == pos, 1, -1)
  regr.task = makeRegrTask(
    id = stri_paste(getTaskId(.task), pos, sep = "."),
    data = data,
    target = target.name,
    weights = getTaskWeights(.task),
    blocking = .task$blocking)
  model = train(.learner$next.learner, regr.task, weights = .weights)
  makeChainModel(next.model = model, cl = "ClassificationViaRegressionModel")
}


predictLearner.ClassificationViaRegressionWrapper = function(.learner, .model, .newdata, .subset = NULL, ...) {
  model = getLearnerModel(.model, more.unwrap = FALSE)
  p = predict(model, newdata = .newdata, subset = .subset, ...)$data$response
  
  if (.learner$predict.type == "response") {
    factor(ifelse(p > 0, getTaskDesc(.model)$positive, getTaskDesc(.model)$negative))
  } else {
    td = getTaskDesc(.model)
    levs = c(td$positive, td$negative)
    propVectorToMatrix(vnapply(p, function(x) exp(x) / sum(exp(x))), levs)
  }
}


getLearnerProperties.ClassificationViaRegressionWrapper = function(learner) {
  props = getLearnerProperties(learner$next.learner)
  props = union(props, c("twoclass", "prob"))
  intersect(props, mlr$learner.properties$classif)
}


setPredictType.ClassificationViaRegressionWrapper = function(learner, predict.type) {
  assertChoice(predict.type, c("response", "prob"))
  learner$predict.type = predict.type
}


isFailureModel.ClassificationViaRegressionModel = function(model) {
  isFailureModel(getLearnerModel(model, more.unwrap = FALSE))
}

