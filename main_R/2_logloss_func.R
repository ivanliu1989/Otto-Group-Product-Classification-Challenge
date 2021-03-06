# logloss <- function(target,pred){
#     n <- nrow(pred)
#     m <- 9
#     p <- 0
# #     P <- mapply(function(pre,tar){p <- tar * max(min(pre,1-10^-15),10^-15) + p}, pred, target)
#     for(i in 1:n){
#         for(j in 1:m){
#             p <- target[i,j] * max(min(pred[i,j],1-10^-15),10^-15) + p
#         }
#     }
#     return(-1/n*p)
# }

MulLogLoss <- function(actual, predicted, eps=1e-15) {
    predicted[predicted < eps] <- eps;
    predicted[predicted > 1 - eps] <- 1 - eps;
    -1/nrow(actual)*(sum(actual*log(predicted)))
}

LogLoss <- function(actual, predicted, eps=1e-15) {
    predicted[predicted < eps] <- eps;
    predicted[predicted > 1 - eps] <- 1 - eps;
    -1/length(actual)*(sum(actual*log(predicted)))
}

shuffle <- function(sf){
    sf[,'id2'] <- sample(1:nrow(sf), nrow(sf), replace=T)
    sf <- sf[order(sf$id2),]
    sf[,'id2'] <- NULL
    return (sf)
}
    
rangeScale <- function(x){
    (x-min(x))/(max(x)-min(x))
}

center_scale <- function(x) {
    scale(x, scale = FALSE)
}