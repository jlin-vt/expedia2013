## Define data type
classes <- c("integer","Date","integer",
             "integer", "double", "double",
             "integer", "integer", "integer",
             "double", "integer", "double",
             "double", "double", "integer",
             "double", "integer", "integer",
             "integer", "integer", "integer",
             "integer", "integer", "integer", 
             "double", "double", "integer",
             "integer", "integer", "double",
             "integer", "integer", "double",
             "integer", "integer", "double",
             "integer", "integer", "double",
             "integer", "integer", "double",
             "integer", "integer", "double",
             "integer", "integer", "double",
             "integer", "integer", "double",
             "integer", "double", "integer")

## read data
paths <- fromJSON(paste(readLines("config.json"), collapse=""))
train <- fread(paths$train_path, nrows=1000, colClasses=classes, na.strings='NULL')

cat("Print the data basic information: \n")
cat("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
cat(sprintf("It has %s observations and %s features", nrow(train), ncol(train)))
#cat("It has", nrow(train), "observations and", ncol(train), "features. \n ", sep = " ")

cat("Print the data completeness: \n")
cat("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
print(as.list(colMeans(is.na(train))))
cat(" \n")

cat("Print data type: \n")
cat("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
print(lapply(train, typeof))
cat(" \n")

cat("Print the first a few rows of data: \n")
cat("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
print(head(train))
cat(" \n")

cat("Print the information of 1st outcome: \n")
cat("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
cat(sprintf("Booking rate: %s", sum(train$click_bool==1)/nrow(train)))
#cat("Booking rate:", sum(train$click_bool==1)/nrow(train), "\n")
cat(" \n")

cat("Print the information of 2nd outcome: \n")
cat("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
cat(sprintf("Booking rate: %s", sum(train$booking_bool==1)/nrow(train)))
#cat("Booking rate:", sum(train$booking_bool==1)/nrow(train), "\n")
cat(" \n")