library(dplyr)
library(tibble)
library(cmdstanr)
library(ggplot2)
library(bayesplot)


load_data <- function(path) {
  df <- read.csv(path, skip=4)
  df <- df %>%
    select(-c("Country.Name", "Indicator.Name", "Indicator.Code", "X"))
  names(df) <- gsub("^X", "", names(df))
  df
}

root <- "D:/phd/asef/tutorstvo/sdg7/data/"
setwd(root)

pct_electricity <- load_data("API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_4752360/API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_4752360.csv")
prod_nuclear <- load_data("API_EG.ELC.NUCL.ZS_DS2_en_csv_v2_4639709/API_EG.ELC.NUCL.ZS_DS2_en_csv_v2_4639709.csv")
health_expenditure <- load_data("API_SH.XPD.CHEX.GD.ZS_DS2_en_csv_v2_4753887/API_SH.XPD.CHEX.GD.ZS_DS2_en_csv_v2_4753887.csv")
gdp_usd <- load_data("API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4764266/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_4764266.csv")  # Actually cents??
gdp_per_capita_usd <- load_data("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4764339/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4764339.csv")
prod_oil_gas_coal <- load_data("API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_4538179/API_EG.ELC.FOSL.ZS_DS2_en_csv_v2_4538179.csv")
prod_oil <- load_data("API_EG.ELC.PETR.ZS_DS2_en_csv_v2_4694805/API_EG.ELC.PETR.ZS_DS2_en_csv_v2_4694805.csv")
prod_renewable_excl_hydroelectric <- load_data("API_EG.ELC.RNWX.KH_DS2_en_csv_v2_4749358/API_EG.ELC.RNWX.KH_DS2_en_csv_v2_4749358.csv")

country_code <- "PAK"
years <- 1960:2022

df <- data.frame(year = years)

preprocess <- function(df, colname) {
  tmp_df <- data.frame(t(df %>% filter(Country.Code == country_code) %>% select(any_of(as.character(years)))))
  names(tmp_df) <- colname
  tmp_df$year <- as.integer(rownames(tmp_df))
  tmp_df
}

df <- df %>%
  left_join(preprocess(pct_electricity, "pct_electricity"), by = "year") %>%
  left_join(preprocess(prod_nuclear, "prod_nuclear"), by = "year") %>%
  left_join(preprocess(health_expenditure, "health_expenditure"), by = "year") %>%
  left_join(preprocess(gdp_per_capita_usd, "gdp_per_capita_usd"), by = "year") %>%
  left_join(preprocess(prod_oil_gas_coal, "prod_oil_gas_coal"), by = "year") %>%
  left_join(preprocess(prod_oil, "prod_oil"), by = "year") %>%
  left_join(preprocess(prod_renewable_excl_hydroelectric, "prod_renewable_excl_hydroelectric"), by = "year") %>%
  left_join(preprocess(gdp_usd, "gdp_mil_usd") %>% mutate(gdp_mil_usd = gdp_mil_usd / 1000000000), by = "year") %>%
  
  mutate(pct_electricity = pct_electricity / 100.0) %>%  # To percentage
  mutate(gdp_per_capita_usd_standardized = (gdp_per_capita_usd - mean(gdp_per_capita_usd, na.rm = T)) / sd(gdp_per_capita_usd, na.rm = T)) %>%  # Standardize
  mutate(prod_renewable_excl_hydroelectric = (prod_renewable_excl_hydroelectric - mean(prod_renewable_excl_hydroelectric, na.rm = T)) / sd(prod_renewable_excl_hydroelectric, na.rm = T))  # Standardize

remove_na <- function(x) {
  mask <- !is.na(x)
  x[mask]
}

fill_na <- function(x) {
  y <- x
  mask <- is.na(y)
  y[mask] <- 0.0
  y
}

to_vec <- function(x) {
  c(unlist(unname(x)))
}

feature_names <- c("prod_nuclear", "health_expenditure", "prod_renewable_excl_hydroelectric")

df_subset <- df %>% 
  select(feature_names) %>%
  mutate(across(where(is.numeric), scale, center = TRUE))
m <- ncol(df_subset)
n <- nrow(df_subset)

target <- df %>% select(pct_electricity)

stan_data <- list(
  n = n,
  m = m,
  o_y = fill_na(to_vec(target)),
  mask_y = (1:n)[is.na(to_vec(target))],
  n_y = sum(!is.na(to_vec(target)))
)

for (i in 1:m) {
  new_list <- list(
    fill_na(to_vec(df_subset[1:n, i])),
    (1:n)[is.na(to_vec(df_subset[1:n, i]))],
    sum(!is.na(to_vec(df_subset[1:n, i])))
  )
  names(new_list) <- c(
    sprintf("o_f%s", i),
    sprintf("mask_f%s", i),
    sprintf("n_f%s", i)
  )
  stan_data <- c(stan_data, new_list)
}


model <- cmdstan_model("../single_country_reparameterized.stan")
fit <- model$sample(
  data = stan_data,
  parallel_chains = 4,
  seed = 1
)

mcmc_areas(fit$draws("weights"))

