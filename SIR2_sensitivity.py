import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import numpy as tfp
tfd = tfp.distributions
import scipy as scp
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


data = pd.read_csv('/Bayesian Statistics/Project/Data/archive/covid_de.csv')
print(data.head)

data2 = data.groupby("date").aggregate("sum")
#print(data2.shape)
#print(data2.head())
#print(data2.index[0:30])

data2 = data2[19:]

from datetime import datetime, date
date_0 = datetime.fromisoformat(data2.index[0])

def relative_day(date):
    time_delta = datetime.fromisoformat(date) - date_0
    return time_delta.days


dates = data2.index.to_series().apply(lambda x: relative_day(x)).values
cases = data2["cases"].values
#plt.plot(cases)
#plt.show()

smoothed_cases = cases.copy()
window = 7
for i in range(1,len(smoothed_cases)):

  if i<window:
    smoothed_cases[i] = np.round(np.mean(cases[:i]))
  else:
    smoothed_cases[i] = np.round(np.mean(cases[(i-window):i]))


infected = np.cumsum(smoothed_cases)
population = 83000000
susceptible = population - infected

for i in range(len(smoothed_cases)):
    if i>=14:
        infected[i] -= np.sum(smoothed_cases[:(i-14)])

removed = population - infected - susceptible
infected = infected.astype("float")
removed = removed.astype("float")
susceptible = susceptible.astype("float")
#plt.plot(infected)
#plt.plot(removed)
#plt.plot(susceptible)
#plt.show()


infected = pd.Series(infected[6:113])
removed = pd.Series(removed[6:113])
susceptible = pd.Series(susceptible[6:113])

#plt.plot(infected[0])
#plt.show()

print(infected)
print(removed)
print(susceptible)

N_days = len(infected)
#phi=0.995

diff_s = susceptible - susceptible.shift(-1)


def issorted_strict(l, T):
    n = len(l)
    if (l[0]<=0 or l[n-1]>=T):
        return False
    for i in range(n-1):
        if l[i]>=l[i+1]:
            return False
    return True

def hybrid_sampler(n_iter=10, lambda_size=3, phi=0.995, alphas=3*[2], betas=3*[4], a=1, b=1, sigma=0.02, M=1):

  p_sample = np.zeros(n_iter)
  lambda_sample = np.zeros((n_iter, lambda_size))
  t_sample = np.zeros((n_iter, lambda_size-1))

  p_sample[0] = a/(a+b)
  lambda_sample[0] = np.array(alphas)/np.array(betas)
  t_sample[0] = np.ceil(N_days*np.arange(1,lambda_size)*1/lambda_size)

  s0 = susceptible[0]
  sT = susceptible[N_days-1]
  i0 = infected[0]
  iT = infected[N_days-1]
  sum_i = np.sum(infected)

  acc_lambda = 0
  acc_t = 0


  for iter in range(1,n_iter):

    if iter%100==0: print(iter)
    # gibbs sampling for p_i->r
    p_sample[iter] = tfd.Beta(s0-sT+i0-iT+a, sT-s0+sum_i-i0+b).sample(1)


    for j in range(lambda_size):
      # metropolis-hastings fot lambda

      time_vec = np.concatenate(([0], t_sample[iter-1], [N_days-1]))

      new_lambda = lambda_sample[iter-1].copy()
      epsilon = sigma*tfd.Normal(0,1).sample(1)
      while new_lambda[j] + epsilon < 0:
        epsilon = sigma*tfd.Normal(0,1).sample(1)
      new_lambda[j] += epsilon

      t_prop = np.arange(time_vec[j], time_vec[j+1])

      kappa_prop = (1/phi-1)*susceptible[t_prop]*(1-np.exp(-new_lambda[j]*infected[t_prop]/population))
      kappa = (1/phi-1)*susceptible[t_prop]*(1-np.exp(-lambda_sample[iter-1][j]*infected[t_prop]/population))


      lambda_logratio = np.sum(scp.special.loggamma(diff_s[t_prop] + kappa_prop) - scp.special.loggamma(diff_s[t_prop] + kappa) - scp.special.loggamma(kappa_prop) + scp.special.loggamma(kappa) + (kappa_prop - kappa) * np.log(1 - phi))
      lambda_logratio += tfd.Gamma(alphas[j], betas[j]).log_prob(new_lambda[j]) - tfd.Gamma(alphas[j], betas[j]).log_prob(lambda_sample[iter-1,j])

      alphaLambda = np.min([1, np.exp(lambda_logratio)])

      if np.random.rand() < alphaLambda:
        lambda_sample[iter, j] = new_lambda[j]
        acc_lambda = acc_lambda + 1;
      else:
        lambda_sample[iter, j] = lambda_sample[iter-1, j]



    for j in range(lambda_size-1):
    # metropolis-hastings for t

      new_t = t_sample[iter-1,:].copy()
      if j>0:
        new_t[0:j] = t_sample[iter, 0:j]

      epsilon = (1-2*(np.random.rand()<0.5))*np.random.randint(1,M+1)
      new_t_temp = new_t.copy()
      new_t_temp[j] += epsilon
      while not(issorted_strict(new_t_temp, N_days)):
        epsilon = (1-2*(np.random.rand()<0.5))*np.random.randint(1,M+1)
        new_t_temp[j] += epsilon
      curr_t = new_t.copy()
      new_t[j] = new_t[j] + epsilon


      if epsilon < 0:
        t_prop_vect = np.arange(new_t[j],curr_t[j]).astype("int")
        lambda_prop = lambda_sample[iter, j+1]
        old_lambda = lambda_sample[iter, j]
      else:
        t_prop_vect = np.arange(curr_t[j],new_t[j]).astype("int")
        lambda_prop = lambda_sample[iter, j]
        old_lambda = lambda_sample[iter, j+1]

      kappa_prop = (1/phi-1)*susceptible[t_prop_vect]*(1-np.exp(-lambda_prop*infected[t_prop_vect]/population))
      kappa = (1/phi-1)*susceptible[t_prop_vect]*(1-np.exp(-old_lambda*infected[t_prop_vect]/population))

      t_logratio = np.sum(scp.special.loggamma(diff_s[t_prop_vect] + kappa_prop) - scp.special.loggamma(diff_s[t_prop_vect] + kappa) - scp.special.loggamma(kappa_prop) + scp.special.loggamma(kappa) + (kappa_prop - kappa) * np.log(1 - phi))

      alphat = np.min([1, np.exp(t_logratio)])

      if np.random.rand() < alphat:
        t_sample[iter, j] = new_t[j]
        acc_t = acc_t + 1;
      else:
        t_sample[iter, j] = t_sample[iter-1,j]

  return(p_sample, lambda_sample, t_sample, acc_lambda/n_iter/lambda_size, acc_t/n_iter/(lambda_size-1))

def phi_sens():

  burn_in = 2000#3500
  lambda_size=3
  alphas=lambda_size*[2]
  betas=lambda_size*[4]
  
  phis = [0.9, 0.95, 0.99, 0.995]
  n_iter = 12000

  # Plot lambda
  for phi in phis:
    phi = phi
    p_sample, lambda_sample, t_sample, acc_lambda, acc_t = hybrid_sampler(n_iter=n_iter, M=1, lambda_size=lambda_size, alphas=alphas, betas=betas)
    np.save("SIRmodel_sens/p_ir_sample_phi"+str(phi)+".npy", p_sample)
    np.save("SIRmodel_sens/lambda_sample_phi"+str(phi)+".npy", lambda_sample)
    np.save("SIRmodel_sens/t_sample_phi"+str(phi)+".npy", t_sample)

  fig, ax = plt.subplots(lambda_size, 2)
  legend1 = []
  for phi in phis:
    #p_sample = np.load("SIRmodel_sens/p_ir_sample_phi"+str(phi)+".npy")
    lambda_sample = np.load("SIRmodel_sens/lambda_sample_phi"+str(phi)+".npy")
    #t_sample = np.load("SIRmodel_sens/t_sample_phi"+str(phi)+".npy")
    print(lambda_sample)
    legend_ = '$\phi = $' + str(phi)
    legend1.append(legend_)

    for i in range(0, lambda_size):
      ax[i,0].plot(lambda_sample[burn_in:,i-1], alpha=0.7, linewidth=0.5)
      ax[i,1].hist(lambda_sample[burn_in:,i-1], bins=80, alpha=0.7)
  fig.legend(legend1, loc="lower center", ncol=4)
  fig.suptitle('Simulation results for $\lambda$ - Varying $\phi$')


  # Plot p
  fig2, ax2 = plt.subplots(1,2)#, figsize=(24,4))
  legend2 = []
  for phi in phis:
    p_sample = np.load("SIRmodel_sens/p_ir_sample_phi"+str(phi)+".npy")
    #lambda_sample = np.load("SIRmodel_sens/lambda_sample_phi"+str(phi)+".npy")
    #t_sample = np.load("SIRmodel_sens/t_sample_phi"+str(phi)+".npy")
    print(p_sample)
    legend_ = '$\phi = $' + str(phi)
    legend2.append(legend_)
    ax2[0].plot(p_sample[burn_in:], alpha=0.7, linewidth=0.5)
    ax2[0].set_xlabel('Iteration')
    ax2[0].set_ylabel('$p^{ir}$')

    ax2[1].hist(p_sample[burn_in:], bins=100, alpha=0.7)
    ax2[1].set_xlabel('Value')
    ax2[1].set_ylabel('Count')

  fig2.legend(legend2, loc="lower center", ncol=4)
  fig2.suptitle('Simulation results for $p^{ir}$ - Varying $\phi$')


  # Plot t
  fig3, ax = plt.subplots(lambda_size-1, 2)
  legend3 = []
  for phi in phis:
    #p_sample = np.load("SIRmodel_sens/p_ir_sample_phi"+str(phi)+".npy")
    #lambda_sample = np.load("SIRmodel_sens/lambda_sample_phi"+str(phi)+".npy")
    t_sample = np.load("SIRmodel_sens/t_sample_phi"+str(phi)+".npy")
    print(t_sample)
    legend_ = '$\phi = $' + str(phi)
    legend3.append(legend_)

    for i in range(0, lambda_size-1):
      ax[i,0].plot(t_sample[burn_in:,i-1], alpha=0.7, linewidth=0.5)
      ax[i,1].hist(t_sample[burn_in:,i-1], bins=80, alpha=0.7)
  fig3.legend(legend3, loc="lower center", ncol=4)
  fig3.suptitle('Simulation results for $t$ - Varying $\phi$')

  plt.show()

def a_sens():

  burn_in = 2000#3500
  lambda_size=3
  alphas=lambda_size*[2]
  betas=lambda_size*[4]
  #betas = [10, 5, 2.5]
  
  a_list = [1, 2, 5, 10]
  n_iter = 12000

  # Plot lambda
  for a in a_list:
    p_sample, lambda_sample, t_sample, acc_lambda, acc_t = hybrid_sampler(n_iter=n_iter, M=1, lambda_size=lambda_size, alphas=alphas, betas=betas, a=a)
    np.save("SIRmodel_sens/p_ir_sample_a"+str(a)+".npy", p_sample)
    np.save("SIRmodel_sens/lambda_sample_a"+str(a)+".npy", lambda_sample)
    np.save("SIRmodel_sens/t_sample_a"+str(a)+".npy", t_sample)

  fig, ax = plt.subplots(lambda_size, 2)
  legend1 = []
  for a in a_list:
    #p_sample = np.load("SIRmodel_sens/p_ir_sample_phi"+str(phi)+".npy")
    lambda_sample = np.load("SIRmodel_sens/lambda_sample_a"+str(a)+".npy")
    #t_sample = np.load("SIRmodel_sens/t_sample_phi"+str(phi)+".npy")
    print(lambda_sample)
    legend_ = '$a = $' + str(a)
    legend1.append(legend_)

    for i in range(0, lambda_size):
      ax[i,0].plot(lambda_sample[burn_in:,i-1], alpha=0.7, linewidth=0.5)
      ax[i,1].hist(lambda_sample[burn_in:,i-1], bins=80, alpha=0.7)
  fig.legend(legend1, loc="lower center", ncol=4)
  fig.suptitle('Simulation results for $\lambda$ - Varying $a$')


  # Plot p
  fig2, ax2 = plt.subplots(1,2)#, figsize=(24,4))
  legend2 = []
  for a in a_list:
    p_sample = np.load("SIRmodel_sens/p_ir_sample_a"+str(a)+".npy")
    #lambda_sample = np.load("SIRmodel_sens/lambda_sample_phi"+str(phi)+".npy")
    #t_sample = np.load("SIRmodel_sens/t_sample_phi"+str(phi)+".npy")
    print(p_sample)
    legend_ = '$a = $' + str(a)
    legend2.append(legend_)
    ax2[0].plot(p_sample[burn_in:], alpha=0.7, linewidth=0.5)
    ax2[0].set_xlabel('Iteration')
    ax2[0].set_ylabel('$p^{ir}$')

    ax2[1].hist(p_sample[burn_in:], bins=100, alpha=0.7)
    ax2[1].set_xlabel('Value')
    ax2[1].set_ylabel('Count')

  fig2.legend(legend2, loc="lower center", ncol=4)
  fig2.suptitle('Simulation results for $p^{ir}$ - Varying $a$')


  # Plot t
  fig3, ax = plt.subplots(lambda_size-1, 2)
  legend3 = []
  for a in a_list:
    #p_sample = np.load("SIRmodel_sens/p_ir_sample_phi"+str(phi)+".npy")
    #lambda_sample = np.load("SIRmodel_sens/lambda_sample_phi"+str(phi)+".npy")
    t_sample = np.load("SIRmodel_sens/t_sample_a"+str(a)+".npy")
    print(t_sample)
    legend_ = '$a = $' + str(a)
    legend3.append(legend_)

    for i in range(0, lambda_size-1):
      ax[i,0].plot(t_sample[burn_in:,i-1], alpha=0.7, linewidth=0.5)
      ax[i,1].hist(t_sample[burn_in:,i-1], bins=80, alpha=0.7)
  fig3.legend(legend3, loc="lower center", ncol=4)
  fig3.suptitle('Simulation results for $t$ - Varying $a$')

  plt.show()

def b_sens():

  burn_in = 2000#3500
  lambda_size=3
  alphas=lambda_size*[2]
  betas=lambda_size*[4]
  #betas = [10, 5, 2.5]
  
  b_list = [1, 2, 5, 10]
  n_iter = 12000

  # Plot lambda
  for b in b_list:
    p_sample, lambda_sample, t_sample, acc_lambda, acc_t = hybrid_sampler(n_iter=n_iter, M=1, lambda_size=lambda_size, alphas=alphas, betas=betas, b=b)
    np.save("SIRmodel_sens/p_ir_sample_b"+str(b)+".npy", p_sample)
    np.save("SIRmodel_sens/lambda_sample_b"+str(b)+".npy", lambda_sample)
    np.save("SIRmodel_sens/t_sample_b"+str(b)+".npy", t_sample)

  fig, ax = plt.subplots(lambda_size, 2)
  legend1 = []
  for b in b_list:
    #p_sample = np.load("SIRmodel_sens/p_ir_sample_phi"+str(phi)+".npy")
    lambda_sample = np.load("SIRmodel_sens/lambda_sample_b"+str(b)+".npy")
    #t_sample = np.load("SIRmodel_sens/t_sample_phi"+str(phi)+".npy")
    print(lambda_sample)
    legend_ = '$b = $' + str(b)
    legend1.append(legend_)

    for i in range(0, lambda_size):
      ax[i,0].plot(lambda_sample[burn_in:,i-1], alpha=0.7, linewidth=0.5)
      ax[i,1].hist(lambda_sample[burn_in:,i-1], bins=80, alpha=0.7)
  fig.legend(legend1, loc="lower center", ncol=4)
  fig.suptitle('Simulation results for $\lambda$ - Varying $b$')


  # Plot p
  fig2, ax2 = plt.subplots(1,2)#, figsize=(24,4))
  legend2 = []
  for b in b_list:
    p_sample = np.load("SIRmodel_sens/p_ir_sample_b"+str(b)+".npy")
    #lambda_sample = np.load("SIRmodel_sens/lambda_sample_phi"+str(phi)+".npy")
    #t_sample = np.load("SIRmodel_sens/t_sample_phi"+str(phi)+".npy")
    print(p_sample)
    legend_ = '$b = $' + str(b)
    legend2.append(legend_)
    ax2[0].plot(p_sample[burn_in:], alpha=0.7, linewidth=0.5)
    ax2[0].set_xlabel('Iteration')
    ax2[0].set_ylabel('$p^{ir}$')

    ax2[1].hist(p_sample[burn_in:], bins=100, alpha=0.7)
    ax2[1].set_xlabel('Value')
    ax2[1].set_ylabel('Count')

  fig2.legend(legend2, loc="lower center", ncol=4)
  fig2.suptitle('Simulation results for $p^{ir}$ - Varying $b$')


  # Plot t
  fig3, ax = plt.subplots(lambda_size-1, 2)
  legend3 = []
  for b in b_list:
    #p_sample = np.load("SIRmodel_sens/p_ir_sample_phi"+str(phi)+".npy")
    #lambda_sample = np.load("SIRmodel_sens/lambda_sample_phi"+str(phi)+".npy")
    t_sample = np.load("SIRmodel_sens/t_sample_b"+str(b)+".npy")
    print(t_sample)
    legend_ = '$b = $' + str(b)
    legend3.append(legend_)

    for i in range(0, lambda_size-1):
      ax[i,0].plot(t_sample[burn_in:,i-1], alpha=0.7, linewidth=0.5)
      ax[i,1].hist(t_sample[burn_in:,i-1], bins=80, alpha=0.7)
  fig3.legend(legend3, loc="lower center", ncol=4)
  fig3.suptitle('Simulation results for $t$ - Varying $b$')

  plt.show()

#phi_sens()
#a_sens()
b_sens()




# lambda_mean = np.mean(lambda_sample[burn_in:], axis=0)
# t_mean = np.round(np.mean(t_sample[burn_in:], axis=0))
# p_mean = np.mean(p_sample[burn_in:])
# lambda_mean, t_mean

# fig, ax = plt.subplots(1,2)#, figsize=(24,4))

# #ax[0].plot(p_sample)
# ax[0].plot(p_sample[burn_in:], color='blue', alpha=0.7, linewidth=0.5)
# ax[0].set_xlabel('Iteration')
# ax[0].set_ylabel('$p^{ir}$')
# ax[0].axhline(p_mean, color='red')

# ax[1].hist(p_sample[burn_in:], bins=100, color='blue', alpha=0.7)
# ax[1].legend(['$p^{ir}$'], loc='upper left')
# ax[1].set_xlabel('Value')
# ax[1].set_ylabel('Count')
# fig.suptitle('Simulation results for $p^{ir}$')

# colors = ['blue', 'orange', 'green']


# fig, ax = plt.subplots(1,2)#, figsize=(24,12))

# for i in range(lambda_size-1):
#     ax[0].plot(t_sample[burn_in:,i], color=colors[i], alpha=0.7, linewidth=0.5)
#     ax[0].axhline(t_mean[i], color='red')
#     ax[1].hist(t_sample[burn_in:,i], bins=100, color=colors[i], alpha=0.7)
# ax[0].set_xlabel('Iteration')
# ax[0].set_ylabel('$t$')
# ax[1].legend(['$t_{1}$'], loc='upper left')
# ax[1].set_xlabel('Value')
# ax[1].set_ylabel('Count')
# fig.suptitle('Simulation results for $t$')


# fig, ax = plt.subplots(lambda_size+1, 2)

# #ax[0,0].plot(lambda_sample)
# #ax[0,0].set_prop_cycle(['red', 'black', 'yellow'])
# for i in range(lambda_size):
#     ax[0,0].plot(lambda_sample[burn_in:,i], color=colors[i], alpha=0.7, linewidth=0.5)
#     ax[0,1].hist(lambda_sample[burn_in:,i], bins=80, color=colors[i], alpha=0.7)

# for i in range(1, lambda_size+1):
#     #ax[i,0].plot(lambda_sample[:,i-1])
#     ax[i,0].plot(lambda_sample[burn_in:,i-1], color=colors[i-1], alpha=0.7, linewidth=0.5)
#     ax[i,0].axhline(lambda_mean[i-1], color='red')
#     #ax[i,1].axhline(lambda_mode2[i-1], color="red")
#     ax[i,1].hist(lambda_sample[burn_in:,i-1], bins=80, color=colors[i-1], alpha=0.7);
#     ax[i,1].axvline(lambda_mean[i-1], color='red')
#     #ax[i,2].axvline(lambda_mode2[i-1], color="red")
# fig.suptitle('Simulation results for $\lambda$')

# plt.show()