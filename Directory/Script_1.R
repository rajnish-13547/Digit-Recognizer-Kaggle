rm(list = ls())
setwd("F:/Kaggle/Digit_Recognizer/Directory")
load(".RData")
library(e1071)

label<- train$label
train$label<- NULL
all<- rbind(train,test)

x_train<-train[,-1]
y_train<-train$label

library("neuralnet")
library(readr)
x_train_reduced<-x_train/255
x_train_cov<-cov(x_train_reduced)
pca_all<-prcomp(x_train_cov)
library(nnet)
response<-class.ind(response)

Xfinal <- as.matrix(x_train_reduced) %*% pca_all$rotation[,1:45]
finalseed <- 150
 set.seed(finalseed)

model_final <- nnet(Xfinal,Y,size=150,softmax=TRUE,maxit=130,MaxNWts = 80000)

#--------------------------------------------------------------------------------
# SVM Analysis
#--------------------------------------------------------------------------------
label<- train$label
train$label<- NULL
apply(train,2,var)
pca <- prcomp(train)
biplot(pca,scale =0)

pca$rotation=-pca$rotation
pca$x=-pca$x
biplot (pca , scale =0)

train.data <- predict(pca,train)
test.data <- predict(pca, test)
train.data <- as.data.frame(train.data)
test.data <- as.data.frame(test.data)
train.data <- train.data[,1:50]
test.data <- test.data[,1:50]

library(e1071)
svm_model <- svm(label~.,train.data)
pred_svm <- predict(svm_model,test.data)
predictions <- data.frame(ImageId=1:nrow(test), Label=pred_svm)

write.csv(predictions, "nnet.csv")