df <- read.csv(".\\data\\student-data.csv")
df_clean <- df[,c(1,12,13,15,16)]
df_clean <- na.omit(df_clean)
# Do some cleaning on chr columns
df_clean$STUD_NO_ANONYMOUS <- trimws(df_clean$STUD_NO_ANONYMOUS)
df_clean$CRS_DPT_CD <- trimws(df_clean$CRS_DPT_CD)
df_clean$HDR_CRS_LTTR_GRD <- trimws(df_clean$HDR_CRS_LTTR_GRD)

# Factor grades column
grades <-
  c("A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F")
df_clean$HDR_CRS_LTTR_GRD <-
  factor(df_clean$HDR_CRS_LTTR_GRD, levels = grades)

# Create course code column
df_clean$COURSE_CODE <- paste(df_clean$CRS_DPT_CD, df_clean$CRS_NO, sep = ".")
df_clean <- df_clean[,-(2:3)]
df <- subset(df, HDR_CRS_PCT_GRD < 999)
df_clean <- subset(df_clean, HDR_CRS_PCT_GRD < 999)
df_clean


## TODO
# remove all students which fail out... all grades == 0
# soln: sum total grade % for each student, remove all equal to 0


# should we include those withdraws?....

N <- nrow(df_clean)
trainID <- sample(N, size = N*0.8, replace = FALSE)
testID <- setdiff(1:N, trainID)

x_train <- df_clean[trainID,4]
x_test <- df_clean[testID,4]
y_train <- df_clean[trainID,2]
y_test <- df_clean[testID,2]


library(xgboost)

xgb_model <- xgboost(data = as.matrix(x_train),
                     label = y_train,
                     objective = "reg:squarederror",
                     colsample_bytree = 0.3,
                     learning_rate = 0.1,
                     max_depth = 5,
                     alpha = 10,
                     nrounds = 100)

y_pred <- predict(xgb_model, as.matrix(x_test))

rmse <- sqrt(mean((y_test - y_pred)^2))
print(paste("Root Mean Squared Error:", rmse))
