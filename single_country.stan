functions {
  
  real alpha_function(real y_prev, real f1, real alpha_bias, real alpha_w0, real alpha_w1) {
    return exp(alpha_bias + alpha_w0 * y_prev + alpha_w1 * f1);
  }
  
  real beta_function(real y_prev, real f1, real beta_bias, real beta_w0, real beta_w1) {
    return exp(beta_bias + beta_w0 * y_prev + beta_w1 * f1);
  }
  
}

data {
  int<lower=0> n;  // number of time steps
  
  int<lower=0> n_obs_y;  // number of observed steps with target
  int<lower=1, upper=n> idx_y_observed[n_obs_y];  // indices of observed steps with target
  int<lower=1, upper=n> idx_y_missing[n - n_obs_y];  // indices of missing steps with target
  vector<lower=0, upper=1>[n_obs_y] y_observed;  // target
  
  int<lower=0> n_obs_f1;  // number of observed steps with feature 1
  int<lower=1, upper=n> idx_f1_observed[n_obs_f1];  // indices of observed steps with feature 1
  int<lower=1, upper=n> idx_f1_missing[n - n_obs_f1];  // indices of observed steps with feature 1
  vector[n_obs_f1] f1_observed;  // feature 1
}

parameters {
  // Target
  vector<lower=0, upper=1>[n - n_obs_y] y_missing;
  real<lower=0, upper=1> y0;
  
  // Feature 1
  vector[n - n_obs_f1] f1_missing;
  real f1_drift;
  real<lower=0> f1_sigma;
  real f1_0;
  
  // Model parameters
  real alpha_bias;
  real alpha_w0;
  real alpha_w1;
  
  real beta_bias;
  real beta_w0;
  real beta_w1;
}

transformed parameters {
  // Target
  vector<lower=0, upper=1>[n] y;
  y[idx_y_observed] = y_observed;
  y[idx_y_missing] = y_missing;
  
  
  // Feature 1
  vector[n] f1;
  f1[idx_f1_observed] = f1_observed;
  f1[idx_f1_missing] = f1_missing;
}

model {
  // Feature 1 prior (some basics + brownian motion)
  f1_drift ~ normal(0, 1);
  f1_sigma ~ cauchy(0, 5);
  f1_0 ~ normal(0, 1);  // feature assumed standardized
  f1[1] ~ normal(f1_0 + f1_drift, f1_sigma);
  for (i in 2:n) {
    f1[i] ~ normal(f1[i-1] + f1_drift, f1_sigma);
  }
  
  // Target prior
  y0 ~ beta(1, 1);
  
  // Model coefficient priors
  alpha_bias ~ normal(0, 1);
  alpha_w0 ~ normal(0, 1);
  alpha_w1 ~ normal(0, 1);
  
  beta_bias ~ normal(0, 1);
  beta_w0 ~ normal(0, 1);
  beta_w1 ~ normal(0, 1);

  // Likelihood (beta process)
  y[1] ~ beta(alpha_function(y0, f1[1], alpha_bias, alpha_w0, alpha_w1), beta_function(y0, f1[1], beta_bias, beta_w0, beta_w1));
  for (i in 2:n) {
    y[i] ~ beta(alpha_function(y[i-1], f1[i], alpha_bias, alpha_w0, alpha_w1), beta_function(y[i-1], f1[i], beta_bias, beta_w0, beta_w1));
  }
}