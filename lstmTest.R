rm(list=ls())
source("lstmFunctions.R")

load("input/allData_13_15.Rdata")

#für die aggreg von allen wohnungen
dat.s<-apply(allData,1,function(x) mean(x, na.rm=T))


dat.in <- dat.s %>%
  tk_tbl() %>%
  mutate(index = index,
         value = data,
         data=NULL)


time<-as.POSIXct(dat.in$index, format="%d.%m.%Y %H:%M", tz = "GMT")
attributes(time)$tzone <- "Europe/Zurich"
time<-time - 3600
dat.in$index<-time

#no aggregation
# dat.in<-dat.in%>%
#   as_tbl_time(index = index)

#aggregation per hour
dat.in<-dat.in%>%
  as_tbl_time(index = index)%>%
  mutate(index = as.POSIXct(round(index,"hours")))%>%
  group_by(index) %>%
  summarise(value=sum(value,na.rm=T))

#train / test

#...for hour aggregation
periods_train <- 24*7*4*3
periods_test  <- 84
skip_span     <- 24*7*4

focusSplit<-3

# max_lag <- 24*365
# 
# p1<-dat.in %>%
#   tidy_acf(value, lags = 0:max_lag) %>%
#   ggplot(aes(lag, acf)) +
#   geom_segment(aes(xend = lag, yend = 0), color = palette_light()[[1]]) +
#   theme_tq() +
#   labs(title = "ACF: Jahr")
# 
# max_lag <- 24*7*4*3
# 
# p2<-dat.in %>%
#   tidy_acf(value, lags = 0:max_lag) %>%
#   ggplot(aes(lag, acf)) +
#   geom_segment(aes(xend = lag, yend = 0), color = palette_light()[[1]]) +
#   theme_tq() +
#   labs(title = "ACF: Saison")
# 
# max_lag <- 24*7*4
# 
# p3<-dat.in %>%
#   tidy_acf(value, lags = 0:max_lag) %>%
#   ggplot(aes(lag, acf)) +
#   geom_segment(aes(xend = lag, yend = 0), color = palette_light()[[1]]) +
#   theme_tq() +
#   labs(title = "ACF: Monat")
# 
# 
# max_lag <- 24*7
# 
# p4<-dat.in %>%
#   tidy_acf(value, lags = 0:max_lag) %>%
#   ggplot(aes(lag, acf)) +
#   geom_segment(aes(xend = lag, yend = 0), color = palette_light()[[1]]) +
#   theme_tq() +
#   labs(title = "ACF: Woche")
# 
# max_lag <- 24
# 
# p5<-dat.in %>%
#   tidy_acf(value, lags = 0:max_lag) %>%
#   ggplot(aes(lag, acf)) +
#   geom_segment(aes(xend = lag, yend = 0), color = palette_light()[[1]]) +
#   theme_tq() +
#   labs(title = "ACF: Tag")
# 
# max_lag <- nrow(dat.in)
# 
# p6<-dat.in %>%
#   tidy_acf(value, lags = 0:max_lag) %>%
#   ggplot(aes(lag, acf)) +
#   geom_segment(aes(xend = lag, yend = 0), color = palette_light()[[1]]) +
#   theme_tq() +
#   labs(title = "ACF: all")
# 
# grid.arrange(p6,p1,p2,p3,p4,p5,ncol=2)


rolling_origin_resamples <- rolling_origin(
  dat.in,
  initial    = periods_train,
  assess     = periods_test,
  cumulative = FALSE,
  skip       = skip_span
)


rolling_origin_resamples$splits[[focusSplit]] %>%
  plot_split(data=dat.in,expand_y_axis = F) +
  theme(legend.position = "bottom")


split    <- rolling_origin_resamples$splits[[focusSplit]]
split_id <- rolling_origin_resamples$id[[focusSplit]]


df_trn <- training(split)
df_tst <- testing(split)

df <- bind_rows(
  df_trn %>% add_column(key = "training"),
  df_tst %>% add_column(key = "testing")
) %>% 
  as_tbl_time(index = index)

rec_obj <- recipe(value ~ ., df) %>%
  step_sqrt(value) %>%
  step_center(value) %>%
  step_scale(value) %>%
  prep()

df_processed_tbl <- bake(rec_obj, df)

center_history <- rec_obj$steps[[2]]$means["value"]
scale_history  <- rec_obj$steps[[3]]$sds["value"]

c("center" = center_history, "scale" = scale_history)


# Model inputs
lag_setting  <- 24*7 #lag zwischen input (x) und prediction (y)
batch_size   <- periods_test
train_length <- periods_test*200
tsteps       <- 1
epochs       <- 100

# Training Set
lag_train_tbl <- df_processed_tbl %>%
  mutate(value_lag = lag(value, n = lag_setting)) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "training") %>%
  tail(train_length)

x_train_vec <- lag_train_tbl$value_lag
x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))

y_train_vec <- lag_train_tbl$value
y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))

# Testing Set
lag_test_tbl <- df_processed_tbl %>%
  mutate(
    value_lag = lag(value, n = lag_setting)
  ) %>%
  filter(!is.na(value_lag)) %>%
  filter(key == "testing")

x_test_vec <- lag_test_tbl$value_lag
x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))

y_test_vec <- lag_test_tbl$value
y_test_arr <- array(data = y_test_vec, dim = c(length(y_test_vec), 1))


model <- keras_model_sequential()

model %>%
  layer_lstm(units            = 500, 
             input_shape      = c(tsteps, 1), 
             batch_size       = batch_size,
             return_sequences = TRUE, 
             stateful         = TRUE) %>%
  layer_dropout(rate = 0.2) %>% 
  layer_lstm(units            = 500, 
             return_sequences = FALSE, 
             stateful         = TRUE) %>%
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 1)%>%
  layer_activation(activation = "linear")


model %>% 
  compile(loss = 'mae', optimizer = 'adam')

model



for (i in 1:epochs) {
  model %>% fit(x          = x_train_arr, 
                y          = y_train_arr, 
                batch_size = batch_size,
                epochs     = 5, 
                verbose    = 1, 
                shuffle    = FALSE)
  
  model %>% reset_states()
  cat("Epoch: ", i,"\n")
}

save(model,file=paste("output/lstmModel/model_lag",lag_setting,"_batch",batch_size,"_epochs",epochs,".Rdata",sep=""))

# Make Predictions
pred_out <- model %>% 
  predict(x_test_arr, batch_size = batch_size) %>%
  .[,1] 

# Retransform values
pred_tbl <- tibble(
  index   = lag_test_tbl$index,
  value   = (pred_out * scale_history + center_history)^2
)

# Combine actual data with predictions
tbl_1 <- df_trn %>%
  add_column(key = "actual")

tbl_2 <- df_tst %>%
  add_column(key = "actual")

tbl_3 <- pred_tbl %>%
  add_column(key = "predict")


ret <- list(tbl_1, tbl_2, tbl_3) %>%
  reduce(time_bind_rows, index = index) %>%
  arrange(key, index) %>%
  mutate(key = as_factor(key))

calc_rmse(ret)


p1<-ret %>% 
  plot_prediction(id = split_id, alpha = 0.65) +
  theme(legend.position = "none")

p2<-ret %>% 
  plot_prediction(id = split_id, alpha = 0.65,focus=T) +
  theme(legend.position = "bottom")

grid.arrange(p1,p2,ncol=1)
