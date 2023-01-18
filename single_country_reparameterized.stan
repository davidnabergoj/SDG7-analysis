functions {
  real mu(real prev_y, real bias, vector weights, row_vector features) {
    // prev_y ... previous target, a number in (0, 1). 
    // We do an logit transform to make the previous target the base for the new prediction
    return 1 / (
      1 + exp(
        -(log(prev_y + 1e-10) - log(1 - prev_y + 1e-10) + bias + dot_product(features, weights))
      )
    );
  }
}

data {
  int<lower=1> m;  // features
  int<lower=1> n;  // years
  
  // Target
  int<lower=0, upper=n> n_y;
  int<lower=0, upper=n> mask_y[n - n_y];
  vector<lower=0, upper=1>[n] o_y;
  
  // Feature 1
  int<lower=0, upper=n> n_f1;
  int<lower=0, upper=n> mask_f1[n - n_f1];
  vector[n] o_f1;
  
  // Feature 2
  int<lower=0, upper=n> n_f2;
  int<lower=0, upper=n> mask_f2[n - n_f2];
  vector[n] o_f2;
  
  // Feature 3
  int<lower=0, upper=n> n_f3;
  int<lower=0, upper=n> mask_f3[n - n_f3];
  vector[n] o_f3;
}

parameters {
  vector<lower=0>[n] alpha_values;
  real<lower=0> beta_value_0;
  real bias;
  vector[m] weights;
  
  vector<lower=0, upper=1>[n - n_y] m_y;
  vector[n - n_f1] m_f1;
  vector[n - n_f2] m_f2;
  vector[n - n_f3] m_f3;
  
  // Prior on features - drift and variance
  vector[m] prior_means;
  vector[m] prior_drifts;
  vector<lower=0>[m] prior_stds;
}

transformed parameters {
  vector[n] y = o_y;
  y[mask_y] = m_y;
  
  vector[n] f1 = o_f1;
  f1[mask_f1] = m_f1;
  
  vector[n] f2 = o_f2;
  f2[mask_f2] = m_f2;
  
  vector[n] f3 = o_f3;
  f3[mask_f3] = m_f3;
  
  matrix[n, m] A;
  A[1:n, 1] = f1;
  A[1:n, 2] = f2;
  A[1:n, 3] = f3;
}

model {
  // General prior
  alpha_values ~ cauchy(0, 15);
  beta_value_0 ~ cauchy(0, 15);
  weights ~ normal(0, 5);
  prior_means ~ normal(0, 5);
  prior_drifts ~ normal(0, 5);
  prior_stds ~ cauchy(0, 5);
  
  // Prior on features
  f1[1] ~ normal(prior_means[1], prior_stds[1]);
  f2[1] ~ normal(prior_means[2], prior_stds[2]);
  f3[1] ~ normal(prior_means[3], prior_stds[3]);
  for (i in 2:n) {
    f1[i] ~ normal(f1[i-1] + prior_drifts[1], prior_stds[1]);
    f2[i] ~ normal(f2[i-1] + prior_drifts[2], prior_stds[2]);
    f3[i] ~ normal(f3[i-1] + prior_drifts[3], prior_stds[3]);
  }
  
  // Likelihood
  y[1] ~ beta(alpha_values[1], beta_value_0);
  for (i in 2:n) {
    y[i] ~ beta(alpha_values[2], alpha_values[2] / mu(y[i - 1], bias, weights, A[i]) - alpha_values[2]);
  }
}
