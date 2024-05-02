df <- read.csv(".\\data\\student-data.csv")
#df <- df[df$DEGR_PGM_CD == "BSC-O ",]
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

length(unique(df_clean$STUD_NO_ANONYMOUS))

courses <- factor(unique(df_clean$COURSE_CODE))

# initializing dataframe for student grades
student_df <- data.frame(Student_ID = unique(df_clean$STUD_NO_ANONYMOUS))
# adding NA columns
for(c in courses){
  student_df[[c]] <- NA
}

for(row in 1:nrow(student_df)){
  
}

