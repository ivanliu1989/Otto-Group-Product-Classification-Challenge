setwd('H:/Machine_Learning/Otto-Group-Product-Classification-Challenge');
setwd('/Users/ivan/Work_directory/Otto-Group-Product-Classification-Challenge');
setwd('C:/Users/Ivan.Liuyanfeng/Desktop/Data_Mining_Work_Space/Otto-Group-Product-Classification-Challenge')
rm(list=ls());gc()
library(h2o)
source('main/2_logloss_func.R')
load(file='data/target.RData')
load(file='data/raw_data_multi.RData')

localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '8g')
h2o.clusterInfo(localH2O)
train <- as.h2o(localH2O, train, key="train")
test <- as.h2o(localH2O, test, key="test")

independent <- colnames(train[,2:94])
dependent <- "target"
fit <- h2o.gbm(y = dependent, x = independent, data = train, 
               n.trees = 15, interaction.depth = 5,
               n.minobsinnode = 2, shrinkage = 0.01, distribution= "multinomial")

fit <- h2o.deeplearning(y = dependent, x = independent, data = train, 
                        classification=T,activation="Tanh",
                        hidden=c(10,10,10),epochs=12,variable_importances=T)

pred <- h2o.predict(object = fit, newdata = test)