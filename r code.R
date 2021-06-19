############################################################################
##                          House Price Predictions
###############################################################################
# Purpose:
#   To submit a credible prediction to kaggle for house prices based on
#   Ames house price training dataset
#   
# Inputs:
#   - 'C:/Users/craig/Google Drive/201705/0526hous/cache/train.csv'
#   - 'C:/Users/craig/Google Drive/201705/0526hous/cache/test.csv'
# 
# Anatomy:
#   1. Required Packages
#   2. Combine Train/Test
#   3. Analyse the Data
#   4. Engineer Variables
#   5. Train a Forest Model
#   6. Score Forest Model
#   7. Train Linear Model
#   8. Score Linear Model
#   9. Make Submission



###1. Required Packages
#install.packages('dplyr')
#install.packages('caret')
#install.packages('party')


###2. Combine Train/Test
setwd("D:/learn/House Prices kaggle")

ktrain<-read.csv('train.csv')
ktest<-read.csv('test.csv')

#create saleprice variable in test data
ktest$SalePrice<-NA
bs<-rbind(ktrain,ktest)
dim(ktrain)
table(ktrain$YearBuilt)


###3. Analyse the Data

#general summary
dim(bs)
summary(bs)

#show freq counts for all variables where <50 categories in the variable
lapply(bs,
       function(var)
       {
         if(dim(table(var))<20) table(var)#if there <20 items then show table
       }
)
#<--although this helps me get a better sense of the data from a univariate
#perspective, there are many trade offs I'm not sure about and I am basically 
#discarding few variables that clearly have no value. A better approach I think
#would be to use a random forest with pretty full growth then assess variable
#importances after that analysis and see if the results can be used to build a
#better linear (or other?) model off the back of it-->


##Push all vars into RandomTreeRegressor to see feature importances

#view mean saleprice by MSSubClass to determine if the variable's continuous 
#property is a true representation
aggregate(data=ktrain,SalePrice~MSSubClass,mean)
#from the data dictionary it does not look like a true representation

#convert MSSubClass to a categorical variable
bs$MSSubClass=as.factor(paste('c',as.character(bs$MSSubClass),sep=''))

#view number of levels for each categorical variable
sapply(bs[,sapply(bs,is.factor)],nlevels)
#there are no factors with >32 levels--OK!

#Replace NA's
nacounts<-sapply(bs, function(x) sum(length(which(is.na(x)))))
types<-sapply(bs,class)
cbind(types,nacounts)


##Loop through list of categoric vars and complete NA's with mode

#define mode2 function (for calculating a variable's mode)
mode2<-function(x){
  ux<-unique(x) 
  ux[which.max(tabulate(match(x,ux)))]
}

#vars to complete with mode
nafacs<-sapply(bs[,sapply(bs,is.factor)], function(x) sum(length(which(is.na(x)))))
data.frame(nafacs)#view results
mvars<-nafacs[0=200 NA's
              dvars<-nafacs[200<=nafacs]#vars with >=200 NA's
              dvarnms<-names(dvars)
              bs[,dvarnms]<-NULL#drop vars
              
              
              ##Loop through numerics completing NA's with median
              
              #vars to complete with mean
              nanums<-sapply(bs[,sapply(bs,is.numeric)], function(x) sum(length(which(is.na(x)))))
              data.frame(nanums)#view results
              mvars<-nanums[00 NAs
                            mvars<-mvars[-12]#exclude saleprice here..
                            
                            #loop through mvars completing NA's with median
                            for(varnm in names(mvars)){
                              print(varnm)
                              bs[,varnm][is.na(bs[,varnm])]<-median(bs[,varnm],na.rm=T)
                            }
                            
                            
                            ##Run the Random Forest to identify best variables
                            
                            #prep data
                            t1<-subset(ktrain,select='Id')
                            tr1<-merge(bs,t1,by='Id')
                            tr1$Id<-NULL
                            
                            #train forest model
                            library(party)
                            mdl_rf<-cforest(SalePrice~.,data=tr1,control=cforest_unbiased(ntree=200))
                            
                            #view feature importances
                            library(dplyr)
                            vi1<-data.frame(varimp(mdl_rf))#get importances
                            vi1<-add_rownames(vi1,'var')#convert rownames to a column
                            names(vi1)[names(vi1)=='varimp.mdl_rf.']<-'varimp'#rename varimp column
                            vi1<-data.frame(vi1[with(vi1,order(-varimp, var)),])
                            vi1#view results
                            
                            
                            
                            ###4. Engineer Variables
                            
                            
                            ##Encode Categoricals
                            
                            #get all factors from bs
                            facs<-data.frame(names(bs)[sapply(bs,is.factor)])
                            names(facs)[1]<-'var'
                            
                            #find the ones that are in the forest's top 20 features
                            merge(facs,vi1[0:20,],by='var')
                            
                            #neighorhood
                            nbhds<-aggregate(data=bs,SalePrice~Neighborhood,mean)
                            names(nbhds)[2]<-'Neighborhood_cg'
                            bs<-merge(x=bs,y=nbhds,by='Neighborhood',all.x=T)
                            
                            #external quality
                            bs$ExterQual_cg<-ifelse(bs$ExterQual=='Ex',5,
                                                    ifelse(bs$ExterQual=='Gd',4,
                                                           ifelse(bs$ExterQual=='TA',3,
                                                                  ifelse(bs$ExterQual=='Fa',2,
                                                                         ifelse(bs$ExterQual=='Po',1,0)))))
                            
                            #basement quality
                            bs$BsmtQual_cg<-ifelse(bs$BsmtQual=='Ex',5,
                                                   ifelse(bs$BsmtQual=='Gd',4,
                                                          ifelse(bs$BsmtQual=='TA',3,
                                                                 ifelse(bs$BsmtQual=='Fa',2,
                                                                        ifelse(bs$BsmtQual=='Po',1,0)))))
                            
                            
                            #kitchen quality
                            bs$KitchenQual_cg<-ifelse(bs$KitchenQual=='Ex',5,
                                                      ifelse(bs$KitchenQual=='Gd',4,
                                                             ifelse(bs$KitchenQual=='TA',3,
                                                                    ifelse(bs$KitchenQual=='Fa',2,
                                                                           ifelse(bs$KitchenQual=='Po',1,0)))))
                            
                            #dwelling type
                            dweltypes<-aggregate(data=bs,SalePrice~MSSubClass,mean)
                            names(dweltypes)[2]<-'MSSubClass_cg'
                            bs<-merge(x=bs,y=dweltypes,by='MSSubClass',all.x=T)
                            
                            #garage interior finish
                            garagefin<-aggregate(data=bs,SalePrice~GarageFinish,mean)
                            names(garagefin)[2]<-'GarageFinish_cg'
                            bs<-merge(x=bs,y=garagefin,by='GarageFinish',all.x=T)
                            
                            #foundation
                            foundations<-aggregate(data=bs,SalePrice~Foundation,mean)
                            names(foundations)[2]<-'Foundation_cg'
                            bs<-merge(bs,foundations,by='Foundation',all.x=T)
                            
                            
                            ##Build New Variables
                            
                            #age at time of sale
                            bs$saleage_cg<-bs$YrSold-bs$YearBuilt
                            
                            #neighbourhood*size
                            bs$nbhdsize_cg<-bs$Neighborhood_cg*bs$GrLivArea
                            
                            
                            
                            ###5. Train a Forest Model
                            
                            
                            ##Define training data
                            
                            #cut back the modified bs data to include only training observations
                            ktrain_id<-subset(ktrain,select='Id')
                            rf1_tr<-merge(bs,ktrain_id,by='Id')
                            use_vars<-names(rf1_tr)
                            rf1_tr<-subset(rf1_tr,select=use_vars)#note that use_vars is re-defined below!!!
                            
                            #split further into train/test for true out of sample test
                            set.seed(1)
                            tr_index<-sample(nrow(rf1_tr),size=floor(.6*nrow(rf1_tr)))
                            rf1_tr_train<-rf1_tr[tr_index,]
                            rf1_tr_test<-rf1_tr[-tr_index,]
                            
                            
                            ##Train cforest model with rf1_tr_train
                            mdl_rf1<-cforest(SalePrice~.,
                                             data=rf1_tr_train,
                                             control=cforest_unbiased(ntree=100, mtry=8, minsplit=30,
                                                                      minbucket=10))
                            
                            #view feature importances
                            vi1<-data.frame(varimp(mdl_rf1))#get importances
                            vi1<-add_rownames(vi1,'var')#convert rownames to a column
                            names(vi1)[names(vi1)=='varimp.mdl_rf1.']<-'varimp'#rename varimp column
                            vi1<-data.frame(vi1[with(vi1,order(-varimp, var)),])
                            vi1#view results
                            
                            #get list of vars that add to model
                            use_vars<-c(vi1[vi1$varimp>0,1][0:30],'SalePrice')
                            use_vars
                            
                            
                            
                            ###6. Score the Forest Model
                            
                            #define log rmse function
                            rmse<-function(x,y){mean((log(x)-log(y))^2)^.5}
                            
                            #define assess function
                            score_rf1<-function(df){
                              preds_rf1<-predict(mdl_rf1,df,OOB=TRUE,type='response')
                              sc1<<-data.frame(cbind(df,preds_rf1))
                              names(sc1)[names(sc1)=='SalePrice.1']<<-'preds_rf1'
                              rmse(sc1$SalePrice,sc1$preds_rf1)
                            }
                            
                            #gauge accuracy
                            score_rf1(rf1_tr_train)#in-sample observations
                            score_rf1(rf1_tr_test)#out of sample observations
                            
                            
                            
                            ###7. Train a Linear Model
                            
                            
                            ##Define training data
                            
                            #cut back the modified bs data to include only training observations
                            ktrain_id<-subset(ktrain,select='Id')
                            lm_tr<-merge(bs,ktrain_id,by='Id')
                            lm_tr$log_SalePrice=log(lm_tr$SalePrice)
                            
                            #subset to include numerics only
                            nums_use<-names(lm_tr)[sapply(lm_tr,is.numeric)]
                            nums_use=c('LotArea','OverallQual','OverallCond','BsmtFinSF1','BsmtUnfSF',
                                       'X1stFlrSF','X2ndFlrSF','Fireplaces','ScreenPorch',
                                       'Neighborhood_cg','KitchenQual_cg','nbhdsize_cg','LotFrontage',
                                       'BsmtFinSF2','BsmtQual_cg','log_SalePrice')
                            
                            #highly correlated features
                            library(caret)
                            write.csv(cor(lm_tr[,nums_use]),file='cor.csv')
                            #<--view correlations in excel-->
                            findCorrelation(cor(lm_tr[,nums_use]))
                            findLinearCombos(lm_tr[,nums_use])
                            lm_tr<-subset(lm_tr,select=nums_use)
                            
                            #split further into train/test for true out of sample test
                            set.seed(1)
                            tr_index<-sample(nrow(lm_tr),size=floor(.6*nrow(lm_tr)))
                            lm_tr_train<-lm_tr[tr_index,]
                            lm_tr_test<-lm_tr[-tr_index,]
                            
                            
                            ##Train a linear model
                            mdl_lm<-lm(log_SalePrice~.,data=lm_tr_train)
                            summary(mdl_lm)#view initial results
                            
                            
                            
                            ###8. Score the Linear Model
                            
                            #define log rmse function
                            rmse<-function(x,y){mean((log(x)-log(y))^2)^.5}
                            
                            #define assess function
                            score_lm<-function(df){
                              preds<-predict(mdl_lm,df,type='response')
                              preds<-exp(preds)
                              df$SalePrice=exp(df$log_SalePrice)
                              sc2<<-data.frame(cbind(df,preds))
                              rmse(sc2$SalePrice,sc2$preds)
                            }
                            
                            #gauge accuracy
                            score_lm(lm_tr_train)#in-sample observations
                            score_lm(lm_tr_test)#out of sample observations
                            
                            
                            
                            ###9. Make Submission
                            
                            #get ktest-altered data
                            t1<-subset(ktest,select='Id')
                            bs_ts<-merge(t1,bs,by='Id')
                            preds<-exp(predict(mdl_lm,bs_ts,OOB=TRUE,type='response'))
                            submit<-data.frame(Id=ktest$Id,SalePrice=preds)
                            write.csv(submit,file='submit.csv',row.names=FALSE)
                            
                            
                            
                            
                            ######Workings
                            