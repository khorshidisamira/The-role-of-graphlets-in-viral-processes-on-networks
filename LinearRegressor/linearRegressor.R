setwd('C:/Users/Samira/Dropbox/Dr_Hasan/Project_graphlet/LinearRegressor')
data=read.csv("FacebookNetwork.csv",header=F)
data=data[,3:41]

data_train=data[1:50,]
data_test=data[51:103,]
model=lm(V41~V4+V7+V10+V11,data=data_train)
yp=predict(model,data_test)
print(mean((yp-data_test$V41)^2)^.5)

model2=lm(V41~V4+V7+V10+V11+V14+V15+V20+V24+V26+V32+V35,data=data_train)
yp2=predict(model2,data_test)
print(mean((yp2-data_test$V41)^2)^.5)


