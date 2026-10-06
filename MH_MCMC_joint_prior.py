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
from scipy.stats import multivariate_normal
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
def log_posterior(theta, data, prior_mean, prior_covariance):
    # theta = [mu, log_sigma]
    
    # Instead of 2 independent priors for each parameter we will use a joint distribution
    
    # This is a tricky situation because me must respect that volatility cant be negative and before in an indepedent priors situation we could use a halfnorm to make sure of that, but here we have to be a bit more technical with the way we go about respecting that volatility bound
    # We can do this by applying log to volatility and then later recovering volatility by using the exponential function e
    
    mu = theta[0]
    log_sigma = theta[1]
    
    sigma = np.exp(log_sigma)
    # Define mean vector and cov matrix
    log_joint_prior = multivariate_normal.logpdf(theta, mean=prior_mean, cov=prior_covariance)
     
    # Compute likelihood with a normal distribution
    #loc = mean, scale = sd
    log_likelihood = np.sum(norm.logpdf(data, loc=mu, scale=sigma))
    
    # Compute posterior
    log_posterior = log_likelihood + log_joint_prior
    return log_posterior

# Define joint priors

# these will be abitrary random geusses of mine 
prior_mean = np.array([0.0004, np.log(0.01)])

prior_sd_mu = 0.0005
prior_sd_log_sigma = 0.05
prior_rho = 0.04

prior_covariance = np.array([
    [
        prior_sd_mu**2,
        prior_rho * prior_sd_mu * prior_sd_log_sigma
    ],
    [
        prior_rho * prior_sd_mu * prior_sd_log_sigma,
        prior_sd_log_sigma**2
    ]
])



# Monte-Carlo settings

num_iter = 5000
#run algo 5000 times
theta_draws = np.zeros((num_iter, 2))
# everytime we produce markov state we want to save it, sos it will be saved here in theta_draws
proposal_sd_mu = 0.0005
proposal_sd_log_sigma = 0.02
# the proposal sd controls the size of jumps between states
proposal_rho = 0.03

# The proposal covariance controls how the alogrithm moves around the parameter space
proposal_cov = np.array([
    [
     proposal_sd_mu**2,
     proposal_rho * proposal_sd_mu * proposal_sd_log_sigma
     ],
    [
     proposal_rho * proposal_sd_mu * proposal_sd_log_sigma,
     proposal_sd_log_sigma**2
     ]
])


burn_in = 1000 
# 1000 states are removed during the burn-in period, we keep 4000 samples



# This is our initial state.
# The chain can start away from the posterior, and after sufficient
# burn-in it should approach the target posterior distribution,
# assuming the Markov chain satisfies the required conditions.
theta_current = np.array([np.mean(returns), np.log(np.std(returns))])


# Run MCMC for a set number of iterations
accepted = 0
for i in range(1,num_iter):
    
    # propose a new state
    theta_proposed = multivariate_normal.rvs(mean=theta_current,cov=proposal_cov)
    
    # Compute acceptance ratio
    # we use exp() to get rid of logs that weve been working with, its an exponential function
    log_alpha = log_posterior(theta_proposed, returns, prior_mean, prior_covariance) - log_posterior(theta_current, returns, prior_mean, prior_covariance)
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
log_sigma_draws = theta_draws[:,1]
sigma_draws = np.exp(log_sigma_draws)

plt.hist(mu_draws, bins=50)
plt.xlabel("Mu")
plt.ylabel("Frequency")
plt.title("Posterior Distribution of Mu")
plt.show()

plt.hist(sigma_draws, bins=50)
plt.xlabel("Sigma")
plt.ylabel("Frequency")
plt.title("Posterior Distribution of Sigma")
plt.show()


plt.plot(mu_draws)
plt.xlabel("Iteration")
plt.ylabel("Mu")
plt.title("Trace Plot - Mu")
plt.show()

plt.plot(sigma_draws)
plt.xlabel("Iteration")
plt.ylabel("Sigma")
plt.title("Trace Plot - Sigma")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(
    mu_draws,
    sigma_draws,
    alpha=0.3
)

plt.xlabel("Mu")
plt.ylabel("Sigma")
plt.title("Joint Posterior Distribution")

plt.show()



   
# compute acceptance rate
acceptance_rate = accepted / (num_iter - 1)   
print(acceptance_rate)


correlation = np.corrcoef(mu_draws, sigma_draws)[0, 1]
print("Posterior correlation:", correlation)


