#Load in required modules
from scipy.stats import beta
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

n_trials_prior =100
n_sucesses_prior = 60
n_trials_likelihood = 100
n_successes_likelihood = 60
n_failure_prior = n_trials_prior - n_sucesses_prior

## x-axis for plotting
numSteps = 1000
x = np.linspace(0, 1, numSteps)

## Lin_successeselihood function
likelihood = []
for i in x:
    event = i**n_successes_likelihood * (1 - i)**(n_trials_likelihood - n_successes_likelihood)
    likelihood.append(event)
## Just normalize lin_successeselihood to integrate to one (for purposes of plotting)
likelihood = likelihood / sum(likelihood) * numSteps
##plot Likelihood
plt.plot(x, likelihood)
##Plot Prior
plt.plot(x, beta.pdf(x, n_sucesses_prior, n_failure_prior))
## Plot posterior
plt.plot(x, beta.pdf(x, n_successes_likelihood + n_sucesses_prior, n_trials_likelihood - n_successes_likelihood + n_failure_prior), linestyle='dashed')
## Legend
plt.legend([f"Likelihood - Binomial({n_trials_likelihood},{n_successes_likelihood})", f"Prior - Beta({n_trials_prior},{n_sucesses_prior})", f"Posterior - Beta"])
