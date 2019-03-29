library(data.table)
library(dplyr)
library(ggplot2)
library(stringr)
library(DT)
library(tidyr)
library(corrplot)
library(leaflet)
library(lubridate)
library(ggthemes)
library(scales)
library(rjson)

source("data_import.R")
source("eda.R")
source("feature_engineering.R")
source("model.R")

## get scores
runscore <- test_result

#' Compute discounted cumulative gain per query.
#' 
#' @param train: data object.
#' @param k: the number of records to consider per query.
#' 
#' @return r: discounted cumulative gain (DCG).
dcg_at_k <- function(dat, k=min(38, nrow(dat))) {
  r1 <- dat$click_bool
  dcg1 <- sum((2 ^ r1 - 1) / log2(2:(length(r1) +1)))
  
  r2 <- dat$booking_bool
  dcg2 <- sum((2 ^ r2 - 1) / log2(2:(length(r2) + 1)))
  
  r <- dcg2 * 4 + dcg1 * 1
  
  return(r)
} 

#' Compute maximum possible discounted cumulative gain for a data set.
#' 
#' @param train: data object.
#' 
#' @return dcg_max: max discounted cumulative gain (DCG)
dcg_max_at_k <- function(dat) {
  dat <- arrange(dat, -booking_bool, -click_bool)
  dcg_max <- dcg_at_k(dat)
  
  return (dcg_max)
  
}

ndcg_at_k <- function(dat) dcg_at_k(dat)/dcg_max_at_k(dat)

## --- test ---
##data <- runscore[, c("SearchId", "click_bool", "booking_bool")] %>% filter(SearchId==8)
##ndcg_at_k(data)

## skip data without click and booking record
index_exc <- 
  runscore %>% 
  group_by(SearchId) %>% 
  summarise(click_bool_sum=sum(click_bool), booking_bool_sum=sum(booking_bool)) %>% 
  filter(click_bool_sum==0 & booking_bool_sum==0) %>% 
  select(SearchId) %>% pull()

index <- setdiff(runscore$SearchId, index_exc)
runscore <- runscore %>% filter(SearchId %in% index)

## run ndcg per query
mean(plyr::ddply(runscore, plyr::.(SearchId), ndcg_at_k)[, 2])

## Remove score and write to file
#test_result <- test_result[, -3]
#write.csv(test_result, file="result.csv", row.names=F)