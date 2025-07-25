
library("randomForest")
library(mclust)
library(cluster)

data <- read.csv("simulated_data.csv")

###
# Unsupervised RF
###

# Run unsupervised random forest
set.seed(123)
rf_model_unsupervised <- randomForest(
  x = data[, setdiff(colnames(data), c("Class", "Subclass"))], 
  y = NULL, ntree = 2000, proximity = TRUE
  )

# PAM
clustering_result <- pam(rf_model_unsupervised$proximity, k = 4)$clustering 

# Save cluster labels
write.csv(clustering_result, "unsupervised_RF_clusters.csv", row.names = FALSE)
