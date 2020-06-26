library(tidytext)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(textdata)
library(data.table)


# Set the working directory
setwd("/Users/singhh/Downloads/CSCI48900")


######### Fairness of Machine Learning Models #########
######### Question 1 #########
loan_data <-read.csv(file="loan_data.csv")

# Split the dataset into training and test
require(caTools)

set.seed(123)   #  set seed to ensure you always have same random numbers generated
sample = sample.split(loan_data,SplitRatio = 0.75) 
train =subset(loan_data,sample ==TRUE) 
test=subset(loan_data, sample==FALSE)

# Convert the variables
train$Default_On_Payment <- as.factor(as.character(train$Default_On_Payment))
train$Age <- as.factor(as.character(train$Age))
train$Credit_History <- as.factor(as.character(train$Credit_History))
train$Marital_Status_Gender <- as.factor(as.character(train$Marital_Status_Gender))
train$Job <- as.factor(as.character(train$Job))
train$Housing <- as.factor(as.character(train$Housing))
train$Dependents <- as.factor(as.character(train$Dependents))

# Convert the variables
test$Age <- as.factor(as.character(test$Age))
test$Credit_History <- as.factor(as.character(test$Credit_History))
test$Marital_Status_Gender <- as.factor(as.character(test$Marital_Status_Gender))
test$Job <- as.factor(as.character(test$Job))
test$Housing <- as.factor(as.character(test$Housing))
test$Dependents <- as.factor(as.character(test$Dependents))

# Get summary
summary(train)

library(mice)
imputed <- complete(mice(train,m=5,maxit=50,meth='pmm',seed=500))

# Build the logistic regression model
imputed <- subset(imputed, select = -c(Customer_ID))
model1 <- glm(Default_On_Payment ~ Age + Credit_History + Marital_Status_Gender + Job + Housing + Dependents, data = imputed, family=binomial)
predict1 <- predict(model1, type="response")


######### Question 1b #########
a <- table(imputed$Default_On_Payment, predict1 >= 0.5)

TP <- a[2,2] # true positives
TN <- a[1,1] # true negatives
FP <- a[1,2] # false positives
FN <- a[2,1] # false negatives

sensitivity <- TP/(TP+FN)
specificity <- TN/(TN+FN)
# Accuracy of the model
accuracy <- (TN + TP)/(TN + TP + FP + FN)

# AUC
library("ROCR") 
pred <- prediction(predict1, imputed$Default_On_Payment)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")     
plot(perf, col=rainbow(7), main="ROC curve Admissions", xlab="Specificity", 
     ylab="Sensitivity")    
abline(0, 1) #add a 45 degree line
auc <- performance(pred, "auc")

# Final prediction
prediction <- predict(model1, newdata=test, type = "response")
solution <- data.frame(Customer_ID = test$Customer_ID, Test_Default_On_Payment = test$Default_On_Payment, Prediction_Default_On_Payment = round(prediction, 0))


######### Question 1c #########
library(data.table)
agebreaks <- c(test$Age)
agelabels <- c("20-30", "30-40", "40-50", "50-60", "60-70",
               "70-80", "80-90", "90-100")


test$age_group <- findInterval(test$Age, seq(0, 100, 10))

solution$Age <- test$Age
solution$age_group <- findInterval(test$Age, seq(0, 100, 10))


# false pos rate group 20-30
solution %>% filter(age_group==1) %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))

# false pos rate group 30-40
solution %>% filter(age_group==2) %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))

# false pos rate group 40-50
solution %>% filter(age_group==3) %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))

# false pos rate group 50-60
solution %>% filter(age_group==4) %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))

# false pos rate group 60-70
solution %>% filter(age_group==5) %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))

# false pos rate group 70-80
solution %>% filter(age_group==6) %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))

# # false pos rate group 80-90
# solution %>% filter(age_group=="7") %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))
# 
# # false pos rate group 90-100
# solution %>% filter(age_group=="8") %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))


######### Random Forest #########
library('randomForest')
# Build the model
model2 <- randomForest(Default_On_Payment ~ Age + Credit_History + Marital_Status_Gender + Job + Housing + Dependents, data = imputed)

# Show model error
plot(model2, ylim=c(0, 0.36))
legend('topright', colnames(model2$err.rate), col=1:3, fill=1:3)

# Get importance
importance    <- importance(model2)
varImportance <- data.frame(Variables = row.names(importance),
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

library('ggthemes')
# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance),
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') +
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() +
  theme_few()

# Predict using the test set
prediction <- predict(model2, test)

# Save the solution
solution2<- data.frame(Customer_ID = test$Customer_ID, Test_Default_On_Payment = test$Default_On_Payment, Prediction_Default_On_Payment = prediction)

solution2$Age <- test$Age
solution2$age_group <- findInterval(test$Age, seq(0, 100, 10))


# false pos rate group 20-30
solution2 %>% filter(age_group==1) %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))

# false pos rate group 30-40
solution2 %>% filter(age_group==2) %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))

# false pos rate group 40-50
solution2 %>% filter(age_group==3) %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))

# false pos rate group 50-60
solution2 %>% filter(age_group==4) %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))

# false pos rate group 60-70
solution2 %>% filter(age_group==5) %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))

# false pos rate group 70-80
solution2 %>% filter(age_group==6) %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))

# # false pos rate group 80-90
# solution2 %>% filter(age_group=="7") %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))
# 
# # false pos rate group 90-100
# solution2 %>% filter(age_group=="8") %>% summarise(sum(Test_Default_On_Payment==0&Prediction_Default_On_Payment==1)/sum(Prediction_Default_On_Payment==1))



################ Conclusion ####################
# The GLM model is not 100% correct because I am getting an accuracy of ~71%
# Also, I tried using different variables in the model and I got different results every single time
# Hence, we can say that the correctness of the model doesn't depend on just one variable like "Age"
# On the other hand, when I used Random Forest, the false positive rates very low compared to GLM model
# This shows that one model could be better than the other in predicting the results



######### Question 2b #########
crime_data <-read.csv(file="compas-scores-two-years-violent.csv")

# Split the dataset into training and test
require(caTools)

set.seed(123)
sample = sample.split(crime_data,SplitRatio = 0.75) 
train =subset(crime_data,sample ==TRUE) 
test=subset(crime_data, sample==FALSE)

# Convert the variables
train$two_year_recid <- as.factor(as.character(train$two_year_recid))
train$sex <- as.factor(as.character(train$sex))
train$race <- as.factor(as.character(train$race))
train$is_recid <- as.factor(as.character(train$is_recid))
train$is_violent_recid <- as.factor(as.character(train$is_violent_recid))
train$score_text <- as.factor(as.character(train$score_text))
train$v_score_text <- as.factor(as.character(train$v_score_text))

# Convert the variables
test$sex <- as.factor(as.character(test$sex))
test$race <- as.factor(as.character(test$race))
test$is_recid <- as.factor(as.character(test$is_recid))
test$is_violent_recid <- as.factor(as.character(test$is_violent_recid))
test$score_text <- as.factor(as.character(test$score_text))
test$v_score_text <- as.factor(as.character(test$v_score_text))



# Get summary
summary(train)

library(mice)
imputed <- complete(mice(train, maxit=0))

# Build the logistic regression model
imputed <- subset(imputed, select = -c(id))

# str(imputed$decile_score.1)

model3 <- glm(two_year_recid ~ sex + race + is_recid + is_violent_recid + score_text + v_score_text, data = imputed, family=binomial)
predict3 <- predict(model3, type="response")


a <- table(imputed$two_year_recid, predict3 >= 0.5)

TP <- a[2,2] # true positives
TN <- a[1,1] # true negatives
FP <- a[1,2] # false positives
FN <- a[2,1] # false negatives

sensitivity <- TP/(TP+FN)
specificity <- TN/(TN+FN)
# Accuracy of the model
accuracy <- (TN + TP)/(TN + TP + FP + FN)

# AUC
library("ROCR") 
pred <- prediction(predict3, imputed$two_year_recid)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")     
plot(perf, col=rainbow(7), main="ROC curve Admissions", xlab="Specificity", 
     ylab="Sensitivity")    
abline(0, 1) #add a 45 degree line
auc <- performance(pred, "auc")

# Final prediction
prediction <- predict(model3, newdata=test, type = "response")
solution3 <- data.frame(id = test$id, test_two_year_recid = test$two_year_recid, prediction_two_year_recid = round(prediction, 0))


######### Question 2a #########

# false pos rate high risk group
solution3 %>% filter(test$score_text=="High") %>% summarise(sum(test_two_year_recid=0&prediction_two_year_recid==1)/sum(prediction_two_year_recid==1))

# false pos rate with respect to some race
solution3 %>% filter(test$race=="Hispanic") %>% summarise(sum(test_two_year_recid=0&prediction_two_year_recid==1)/sum(prediction_two_year_recid==1))



######### Conclusion #########
# The GLM model is not 100% correct but I got a accuracy of ~98% which is so much better from the last experiment
# We can say that the correctness of the model doesn't depend on just one variable like "race"
# But it definitely affects the results in this scenario 

