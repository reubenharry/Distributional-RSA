
library(lmerTest)

df = read.table(file="/Users/reuben/Dropbox/Reuben/Research/Distributional-RSA/dist_rsa/experiment/experiment/experiment/data.csv",sep=",")

df1 = read.table(file="dist_rsa/experiment/experiment/experiment/data1.csv",sep=",")

df2 = read.table(file="dist_rsa/experiment/experiment/experiment/data2.csv",sep=",")

dfJustMets = read.table(file="/Users/reuben/Dropbox/Reuben/Research/Distributional-RSA/dist_rsa/experiment/experiment/experiment/dataJustMets.csv",sep=",")

analysis <- lmer(V4 ~ V3 + (1+V3|V2)+(1+V3|V1), data=df)
summary(analysis)

analysis1 <- lmer(V4 ~ V3 + (1+V3|V2)+(1+V3|V1), data=df1)
summary(analysis1)

analysis2 <- lmer(V4 ~ V3 + (1+V3|V2)+(1+V3|V1), data=df2)
summary(analysis2)

analysisJustMets <- lmer(V4 ~ V3 + (1+V3|V2)+(1+V3|V1), data=dfJustMets)
summary(analysisJustMets)



