
library(plotROC)
g1 <- ggplot() +
  geom_roc(n.cuts=0,data=L2LogicalRegression$pred[selectedIndices, ], mapping=aes(m=normal, d=factor(obs, levels = c("normal", "cancer")))) +
  coord_equal() +
  style_roc()

g2 <- geom_roc(n.cuts=0,data=rpartProbs, mapping=aes(m=normal, d=testing$dx))

g1 + g2 +
  annotate("text", x=0.75, y=0.25, label=paste("Cross-Validation AUC =", round((calc_auc(g1))$AUC, 4))) + 
  style_roc(theme =theme_classic) +
  ggtitle("Logistic Regression ROC") + 
  scale_x_continuous("False Positive Rate", breaks = seq(0, 1, by = .1)) +
  scale_y_continuous("True Positive Rate", breaks = seq(0, 1, by = .1))

getTrainPerf <- function(x) {
  bestPerf <- x$bestTune
  colnames(bestPerf) <- gsub("^\\.", "", colnames(bestPerf))
  out <- merge(x$results, bestPerf)
  out <- out[, colnames(out) %in% x$perfNames, drop = FALSE]
  colnames(out) <- paste("Train", colnames(out), sep = "")
  out$method <- x$method
  out
}