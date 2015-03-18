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

LogLoss <- function(actual, predicted, eps=1e-15) {
    predicted[predicted < eps] <- eps;
    predicted[predicted > 1 - eps] <- 1 - eps;
    -1/nrow(actual)*(sum(actual*log(predicted)))
}