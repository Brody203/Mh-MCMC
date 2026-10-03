# -*- coding: utf-8 -*-
"""
Bayesian Metropolis-Hastings MCMC for SPY Returns

This script estimates the posterior distributions of the mean daily
return (mu) and daily volatility (sigma) of SPY using a
Metropolis-Hastings Markov Chain Monte Carlo algorithm.
"""

import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.stats import halfnorm
import matplotlib.pyplot as plt

# set seed for reproducibility of the MCMC sampling
np.random.seed(42)

# Pull historical SPY market data.
# We model daily log returns as normally distributed:
#
# r_t ~ N(mu, sigma^2)
#
# where mu is the mean daily return and sigma is daily volatility.

data = yf.download(
    "SPY",
    start="2015-01-01",
    end="2026-01-01",
    auto_adjust=True
    )

prices = data['Close']

# log return at time t = ln(price at time t / price at time t - 1)
returns = np.log(prices / prices.shift(1))
returns = returns.dropna()

# 1d return vector
returns = returns.to_numpy().flatten()


# Calculating posterior

# Define target distribution = posterior distribution of our parameters (mean return, volatility)

# theta is a vector of params
def log_posterior(theta, data):
    # theta = [mu, sigma]
    
    
    # Compute prior for each param
    log_prior_mu = norm.logpdf(theta[0], loc=0, scale=0.001)
    
    log_prior_sigma = halfnorm.logpdf(theta[1], loc=0, scale=0.02)
    
    log_prior = log_prior_mu + log_prior_sigma
    
    
    # Compute likelihood with a normal distribution
    #loc = mean, scale = sd
    log_likelihood = np.sum(norm.logpdf(data, loc=theta[0], scale=theta[1]))
    
    # Compute posterior
    log_posterior = log_likelihood + log_prior
    
    return log_posterior

# Monte-Carlo settings

num_iter = 5000
#run algo 5000 times
theta_draws = np.zeros((num_iter, 2))
# everytime we produce markov state we want to save it, sos it will be saved here in theta_draws
proposal_sd_mu = 0.0005
proposal_sd_sigma = 0.0001
# the proposal sd controls the size of jumps between states
burn_in = 1000 
# 1000 states are removed during the burn-in period, we keep 4000 samples



# This is our initial state.
# The chain can start away from the posterior, and after sufficient
# burn-in it should approach the target posterior distribution,
# assuming the Markov chain satisfies the required conditions.
theta_current = np.array([np.mean(returns), np.std(returns)])


# Run MCMC for a set number of iterations
accepted = 0
for i in range(1,num_iter):
    
    # propose a new state
    theta_proposed_mu = norm.rvs(loc=theta_current[0], scale=proposal_sd_mu)
    theta_proposed_sigma = norm.rvs(loc=theta_current[1], scale=proposal_sd_sigma)
    theta_proposed = np.array([theta_proposed_mu, theta_proposed_sigma])
    
    # Compute acceptance ratio
    # we use exp() to get rid of logs that weve been working with, its an exponential function
    log_alpha = log_posterior(theta_proposed, returns) - log_posterior(theta_current, returns)
    if log_alpha >= 0:
        acceptance_ratio = 1
    else:
        acceptance_ratio = np.exp(log_alpha)
    acceptance_ratio = np.minimum(1, acceptance_ratio)
    
    
    # Acceptance/Rejection
    u = np.random.uniform(0,1)
    if u <= acceptance_ratio:
        theta_current = theta_proposed
        accepted += 1
    # store the new state
    theta_draws[i] = theta_current

# now we can remove the bottom 1000 which is our burn-in phase
theta_draws = theta_draws[burn_in:]      


# Plot posterior distribution of mu and sigma
mu_draws = theta_draws[:,0]
sigma_draws = theta_draws[:,1]
plt.hist(mu_draws, bins=50)
plt.show()
plt.hist(sigma_draws, bins=50)
plt.show()


# Trace plot of mu and sigma
plt.plot(mu_draws)
plt.ylim(np.min(mu_draws), np.max(mu_draws))
plt.xlabel("Iteration")
plt.ylabel("Mu")
plt.title("Trace Plot - Mu")
plt.show()

plt.plot(sigma_draws)
plt.xlabel("Iteration")
plt.ylabel("Sigma")
plt.title("Trace Plot - Sigma")
plt.show()
   
# compute acceptance rate
acceptance_rate = accepted / (num_iter - 1)   
print(acceptance_rate)


# even though the priors are calculated independently, the posterior distributuon may still show some dependence between them
plt.figure(figsize=(8, 6))
plt.scatter(mu_draws, sigma_draws, alpha=0.3)
plt.xlabel("Mu")
plt.ylabel("Sigma")
plt.title("Posterior Dependence Between Mu and Sigma")
plt.show()


correlation = np.corrcoef(mu_draws, sigma_draws)[0,1]
print(correlation)

