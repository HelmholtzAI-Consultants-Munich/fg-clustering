
library(datasets)
library(randomForest)
data(iris)

iris.rf <- randomForest(Species ~ ., data=iris, importance=TRUE, proximity=TRUE, ntree=500, mtry=2)
iris.rf$err.rate
iris.rf$proximity