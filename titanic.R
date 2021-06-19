train <- read.csv("C:/Users/hanzu/Downloads/train.csv", stringsAsFactors = FALSE)
test <- read.csv("C:/Users/hanzu/Downloads/test.csv", stringsAsFactors = FALSE)
str(train)
str(test)
train$is_train <- TRUE
test$is_train <- FALSE
names(test)
names(train)
test$Survived <- NA

merged <- rbind(train, test)
head(merged)
tail(merged)
table(merged$is_train)
summary(merged)
table(merged$Embarked)
merged[merged$Embarked=='', "Embarked"] <- 'S'
table(is.na(merged$Age))
median(merged$Age, na.rm = TRUE)
merged[is.na(merged$Age), "Age"] <- 28
median(merged$Fare, na.rm = TRUE)
merged[is.na(merged$Fare), "Fare"] <- 14.4542
table(is.na(merged$Fare))
table(is.na(merged))
str(merged)

merged$Pclass <- as.factor(merged$Pclass)
merged$Sex <- as.factor(merged$Sex)
merged$Age <- as.factor(merged$Age)
merged$SibSp <- as.factor(merged$SibSp)
merged$Parch <- as.factor(merged$Parch)
merged$Embarked <- as.factor(merged$Embarked)

train <- merged[merged$is_train==TRUE,]
test <- merged[merged$is_train==FALSE,]
str(train)
str(test)

train$Survived <- as.factor(train$Survived)

model <- glm(Survived~ Pclass + Sex + SibSp , family = "binomial", data = train)
summary(model)

Survived <- predict(model, newdata = test, type = "response")
Survived <- ifelse(Survived > 0.5, 1, 0)
PassengerId <- test$PassengerId
df <- as.data.frame(PassengerId)
df$Survived <- Survived 
head(df)
getwd()
write.csv(df , file = "titanic_output" , row.names = FALSE )
