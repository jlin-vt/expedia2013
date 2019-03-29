## missing values
missing_values <- train %>% summarize_each(funs(sum(is.na(.))/n()))
missing_values <- gather(missing_values, key="feature", value="missing_pct")
missing_values %>%
  ggplot(aes(x=reorder(feature,-missing_pct),y=missing_pct)) +
  geom_bar(stat="identity",fill="red")+
  coord_flip() + 
  theme_bw()

## cor plot
good_features <- filter(missing_values, missing_pct<0.75)
good_features <- good_features %>% filter(feature!="date_time")
train %>% 
  select(good_features$feature) %>% 
  cor(use="complete.obs") %>% 
  corrplot(type="lower")

## 1d x: discrete
train %>% ggplot(aes(booking_bool)) +
  geom_bar()

## 1d x: continous
train %>%
  ggplot(aes(x=prop_location_score2)) +
  geom_histogram(bins=400, fill="red") +
  ylab("Count") +
  coord_cartesian(x=c(-0.5,0.5)) + 
  theme_bw() + 
  theme(axis.title=element_text(size=16),axis.text=element_text(size=14))

## 1d x: continous (density)
train %>%
  ggplot(aes(visitor_hist_adr_usd)) +
  geom_line(stat="density", color="red", size=1.2)

## 2d x: discrete y: discrete
train %>% ggplot(aes(x=prop_review_score, fill=factor(booking_bool))) +
  geom_bar(stat='count', position='dodge') +
  labs(x='prop_review_score') +
  theme_bw()

## 2d x: discrete(time) y: discrete
train %>%
  mutate(year_month=make_date(year=year(date_time),month=month(date_time))) %>%
  group_by(year_month) %>% summarize(mean_booking_bool=mean(booking_bool)) %>%
  ggplot(aes(x=year_month,y=mean_booking_bool)) +
  geom_line(size=1.5, color="red")+
  geom_point(size=5, color="red")+theme_bw()

## 2d x: discrete(time) y: discrete (see the variation)
train %>%
  mutate(month=month(date_time)) %>%
  group_by(month) %>%
  summarize(mean_booking_bool=mean(abs(booking_bool)),n()) %>%
  ggplot(aes(x=month,y=mean_booking_bool))+
  geom_smooth(color="grey40")+
  geom_point(color="red")+coord_cartesian(ylim=c(0,0.25))+theme_bw()

## 3d x: discrete y: discrete z(grid on): discrete
train %>% ggplot(aes(prop_review_score, fill=factor(booking_bool))) +
  geom_histogram() +
  facet_grid(.~random_bool)

## 3d x: continous y: continous z(grid on): discrete
train %>% 
  group_by(srch_saturday_night_bool, prop_review_score) %>% 
  summarise(mean_booking_bool=mean(booking_bool)) %>% 
  ggplot(aes(x=prop_review_score,y=mean_booking_bool, color=factor(srch_saturday_night_bool))) + 
  geom_point(size=4) +
  theme_fivethirtyeight()

## 2d x: continous y: discrete
train %>% ggplot(aes(x=visitor_hist_adr_usd, fill=as.factor(booking_bool), color=as.factor(booking_bool))) +
  geom_line(stat="density", size=1.2) +
  theme_bw()