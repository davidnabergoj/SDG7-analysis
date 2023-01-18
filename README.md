# SDG7-analysis

## Adding new features

To add new features to a model, you must include them in the definition of the likelihood and the prior.
For the beta process model in `single_county.stan`, this means changing the following function definitions:

```stan
real alpha_function(real y_prev, real f1, real alpha_bias, real alpha_w0, real alpha_w1) {
  return exp(alpha_bias + alpha_w0 * y_prev + alpha_w1 * f1);
} 
real beta_function(real y_prev, real f1, real beta_bias, real beta_w0, real beta_w1) {
  return exp(beta_bias + beta_w0 * y_prev + beta_w1 * f1);
}
```

If we have additional features `f2, f3`, our function definitions would be:
```stan
real alpha_function(real y_prev, real f1, real alpha_bias, real alpha_w0, real alpha_w1) {
  return exp(alpha_bias + alpha_w0 * y_prev + alpha_w1 * f1 + alpha_w2 * f2 + alpha_w3 * f3);
} 
real beta_function(real y_prev, real f1, real beta_bias, real beta_w0, real beta_w1) {
  return exp(beta_bias + beta_w0 * y_prev + beta_w1 * f1 + beta_w2 * f2 + beta_w3 * f3);
}
```

This further requires adding features to the `data` block:
```stan
data {
  ...  // Old code
  
  int<lower=0> n_obs_f2;
  int<lower=1, upper=n> idx_f2_observed[n_obs_f2];
  int<lower=1, upper=n> idx_f2_missing[n - n_obs_f2];
  vector[n_obs_f2] f2_observed; 
  
  int<lower=0> n_obs_f3;
  int<lower=1, upper=n> idx_f3_observed[n_obs_f3];
  int<lower=1, upper=n> idx_f3_missing[n - n_obs_f3];
  vector[n_obs_f3] f3_observed; 
}
```

We need to add new parameters (`alpha_w2, alpha_w3, beta_w2, beta_w3`):

```stan
parameters {
  ... // old code
  
  real alpha_w2;
  real alpha_w3;
  
  real beta_w2;
  real beta_w3;
}
```

The new features should be made into vectors where Stan will run inference on the appropriate missing values:
```stan
transformed parameters {
  ... // old code

  vector[n] f2;
  f2[idx_f2_observed] = f2_observed;
  f2[idx_f2_missing] = f2_missing;
  
  // Same for f3
```

To stabilize model training, we can also add a prior over these features (see below). We then use the features in the likelihood computation:
```stan
model {
  ... // old code

  // Prior for a new feature
  f2_drift ~ normal(0, 1);
  f2_sigma ~ cauchy(0, 5);
  f2_0 ~ normal(0, 1);  // feature assumed standardized
  f2[1] ~ normal(f2_0 + f2_drift, f2_sigma);
  for (i in 2:n) {
    f2[i] ~ normal(f2[i-1] + f2_drift, f2_sigma);
  }
  
  ... // old code
  
  // Updated likelihood
  y[1] ~ beta(
    alpha_function(y0, f1[1], alpha_bias, alpha_w0, alpha_w1),
    beta_function(y0, f1[1], beta_bias, beta_w0, beta_w1)
  );
  
  for (i in 2:n) {
    y[i] ~ beta(
      alpha_function(y[i-1], f1[i], f2[i], f3[i], alpha_bias, alpha_w0, alpha_w1, alpha_w2, alpha_w3), 
      beta_function(y[i-1], f1[i], f2[i], f3[i], beta_bias, beta_w0, beta_w1, beta_w2, beta_w3)
    );
  }
}
```

These changes let a model have additional features, but we still have to provide them to Stan via an interface. 
If you're using `pystan` in Python, you'll likely do this with a pandas dataframe.
If you're using `cmdstanr` in R, you need to pass the correct features to the model as in `electricity_access_analysis.Rmd`:
```r

fit <- model$sample(
  data = list(
    n = nrow(country_data),
    
    n_obs_y = sum(!is.na(country_data$pct_electricity)),
    idx_y_observed = as.array((1:nrow(country_data))[!is.na(country_data$pct_electricity)]),
    idx_y_missing = as.array((1:nrow(country_data))[is.na(country_data$pct_electricity)]),
    y_observed = as.array( remove_na(country_data$pct_electricity)),
    
    n_obs_f1 = sum(!is.na(country_data$gdp_per_capita_usd_standardized)),
    idx_f1_observed = as.array((1:nrow(country_data))[!is.na(country_data$gdp_per_capita_usd_standardized)]),
    idx_f1_missing = as.array((1:nrow(country_data))[is.na(country_data$gdp_per_capita_usd_standardized)]),
    f1_observed = as.array(remove_na(country_data$gdp_per_capita_usd_standardized)),
    
    n_obs_f2 = sum(!is.na(country_data$my_custom_feature)),
    idx_f2_observed = as.array((1:nrow(country_data))[!is.na(country_data$my_custom_feature)]),
    idx_f2_missing = as.array((1:nrow(country_data))[is.na(country_data$my_custom_feature)]),
    f2_observed = as.array(remove_na(country_data$my_custom_feature))
  ),
  parallel_chains = 4,
  seed = 1
)

```

## How to make conclusions

After the Stan run, we can interpret the posterior samples (draws).
Say we plot the histogram of `alpha_w1` draws and `alpha_w2` draws.
The idea is as follows:  
if the first histogram has generally a larger magnitude than the second one, then the first feature contributes more to explaining increased electricity access than the second one.
This is unfortunately not completely true, since the `beta_w1, beta_w2` coefficients must also be observed.

The following would be a decent approach to understanding which feature best explains the target (percentage of people with electricity):  
take the model with a single feature and obtain the posterior draws using N different features. Observe the average log posterior.
The model-feature combination with the highest log posterior value corresponds to the feature that best explains the data.
This will also be apparent from plots of the inferred features - poor models will at best infer something close to the prior (upward trend or downward trend).

The following would be a decent approach to understanding which feature most positively affects the target (as the feature increases, so does the target):  
write the likelihood as a normal distribution whose mean is a linear function of the features and the target at the previous step. Fix the standard deviation to a learnable constant.
Run the model with all features combined. Observe the histograms of linear function coefficients. The more positive the coefficient, the more its feature and the target are positively correlated.
The more negative, the more negatively they are correlated.
This model assumes the target at a given time is taken from a normal distribution. A hack to achieve this is to possibly standardize the target before sending it to the Stan model.
It might be possible to understand this feature-target connection with the beta model as well.
