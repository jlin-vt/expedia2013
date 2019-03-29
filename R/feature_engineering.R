#' Truncate outlier by 0.05 of data as lower bound and 0.95 of data as upper bound.
#' 
#' @param train: data set.
#' @param feature_name: feature name.
#' 
#' @return train: modified data set.
outlier_handler <- function(train, feature_name){
  ub <- quantile(train[feature_name], probs=.95, na.rm=T)
  lb <- quantile(train[feature_name], probs=.05, na.rm=T)
  train[train[feature_name] >ub, feature_name]=ub
  train[train[feature_name] >lb, feature_name]=lb
  
  return(train)
}

#' Impute the feature with with its most correlated feature.
#' 
#' @param train: data set.
#' @param feature_name: feature name.
#' 
#' @return train: modified data set.
impute_with_best_support <- function(train, feature_name){
  ## compute missing value again
  missing_values <- train %>% summarize_each(funs(sum(is.na(.))/n()))
  missing_values <- gather(missing_values, key="feature", value="missing_pct")
  good_features <- filter(missing_values, missing_pct<0.75)
  good_features <- good_features %>% filter(feature != "date_time")
  tmp <- train %>% select(good_features$feature)
  
  ## find support index
  feature_support_name <- cor(tmp, use="complete.obs")[, feature_name] %>% abs() %>% sort(decreasing=T)
  feature_support_name <- names(feature_support_name[2])
  
  feature_support_avg <- mean(train[,feature_support_name], na.rm=T)
  feature_support_std <- sd(train[,feature_support_name], na.rm=T)
  train[is.na(train[feature_name]), feature_name]=rnorm(1, feature_support_avg, feature_support_std)
  
  return(train)
}

## 'prop_review_score'
train <- train %>%
  mutate(prop_review_score=ifelse(is.na(prop_review_score), median(prop_review_score, na.rm=T), prop_review_score))
## 'prop_location_score2'
train <- train %>% mutate(prop_location_score2=replace(prop_location_score2, is.na(prop_location_score2), 0))
## 'visitor_hist_adr_usd'
train <- train %>% mutate(visitor_hist_adr_usd=replace(visitor_hist_adr_usd, is.na(visitor_hist_adr_usd), 0))
## 'visitor_hist_starrating_bool'
train <- train %>%
  mutate(visitor_hist_starrating_bool=ifelse(is.na(visitor_hist_starrating), 0, 1))
## 'date_time'
train <- train %>% mutate(year=year(date_time),month=month(date_time), weekday=wday(date_time))
## prop_log_historical_price
train <- train %>%
  mutate(prop_log_historical_price=ifelse(prop_log_historical_price!=0, 1, prop_log_historical_price))
## srch_query_affinity_score
train <- train %>% mutate(srch_query_affinity_score=replace(srch_query_affinity_score,
                                                            is.na(srch_query_affinity_score),
                                                            median(srch_query_affinity_score, na.rm=T)
)
)
## 'visitor_location_country_id' (find the hottest country visited)
prop_country_id <- train %>% group_by(visitor_location_country_id) %>%
  summarise(n=n()) %>%
  filter(!is.na(n)) %>%
  arrange(desc(n)) %>%
  head(5) %>%
  select(visitor_location_country_id) %>%
  pull()

train <- train %>%
  mutate(visitor_location_country_bool=ifelse(visitor_location_country_id %in% prop_country_id, 1, 0))

## orig_destination_distance
train <- impute_with_best_support(train, 'orig_destination_distance')
train <- outlier_handler(train, 'orig_destination_distance')

##  merge 9 competitors' price info NO.1: comp_rate_sum
for (i in 1:8) {
  feature_name <- paste0('comp', i, '_rate')
  train[is.na(train[feature_name]), feature_name]=0
}

train['comp_rate_sum'] <- train['comp1_rate']
for (i in 2:8) {
  feature_name <- paste0('comp', i, '_rate')
  train['comp_rate_sum'] <- train['comp_rate_sum'] + train[feature_name]
}

##  merge 9 competitors' price info NO.2: comp_rate_inv_sum
for (i in 1:8) {
  feature_name <- paste0('comp', i, '_inv')
  train[is.na(train[feature_name]), feature_name] <- 0
  train[train[feature_name]==1, feature_name] <- 10
  train[train[feature_name]==-1, feature_name] <- 1
  train[train[feature_name]==0, feature_name] <- -1
  train[train[feature_name]==10, feature_name] <- 0
}

train['comp_inv_sum'] <- train['comp1_inv']
for (i in 2:8) {
  feature_name <- paste0('comp', i, '_inv')
  train['comp_rate_sum']=train['comp_rate_sum'] + train[feature_name]
}

## drop useless columns
drop_col <- c()
for (i in 1:8) drop_col <- c(drop_col, paste0('comp', i, '_rate'))
for (i in 1:8) drop_col <- c(drop_col, paste0('comp', i, '_inv'))
for (i in 1:8) drop_col <- c(drop_col, paste0('comp', i, '_rate_percent_diff'))
drop_col <- c(drop_col, 'gross_bookings_usd', 'visitor_hist_starrating',
              'date_time', 'site_id', 'visitor_location_country_id',
              'prop_country_id', 'srch_destination_id', 'position')
train <- train %>% select(-one_of(drop_col))

## what now?
missing_values <- train %>% summarize_each(funs(sum(is.na(.))/n()))
missing_values <- gather(missing_values, key="feature", value="missing_pct")
missing_values %>%
  ggplot(aes(x=reorder(feature,-missing_pct),y=missing_pct)) +
  geom_bar(stat="identity",fill="red")+
  coord_flip()+
  theme_bw()