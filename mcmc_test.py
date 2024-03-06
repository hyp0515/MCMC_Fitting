import numpy as np
import matplotlib.pyplot as plt
import disk_model
import emcee
import corner
np.random.seed(2) # For reproduction

# Model
def T_model(theta):
    a_max, Mstar, Mdot = theta
    opacity_table=disk_model.generate_opacity_table(
        a_min=0,a_max=a_max,q=-3.5,dust_to_gas=0.01
        )
    disk_property_table = disk_model.generate_disk_property_table(
        opacity_table=opacity_table
        )
    DM = disk_model.DiskModel(
        opacity_table=opacity_table, disk_property_table=disk_property_table
        )
    DM.generate_disk_profile(
        Mstar=Mstar*disk_model.Msun, Mdot=Mdot*1e-5*disk_model.Msun/disk_model.yr,
        Rd=30*disk_model.au,Q=1.5,N_R=50
        )
    R_grid = DM.R[1:]/disk_model.au
    T_mid = DM.T_mid
    cut = np.argmax(DM.Sigma > 0)
    R_grid = R_grid[cut:]
    T_mid = T_mid[cut:]
    return R_grid, T_mid

# Synthetic data (Mock Observation)
r_grid, t_mid_fiducial = T_model(theta = (0.1, 0.14, 0.14))
yerr = 50
t_mid_train = t_mid_fiducial + np.random.normal(loc=0.0, scale=yerr, size=len(t_mid_fiducial))
data_indices = np.random.randint(len(r_grid), size=6)
r_data = r_grid[data_indices]
t_data = t_mid_train[data_indices]
# plt.errorbar(r_data, t_data, yerr=yerr, fmt=".k", capsize=0)
# plt.show()


# Log-prob, log-likelihood, and prior
def log_likelihood(theta, r, y, yerr):
    """
    y and r is the temperature and radius data point
    """ 
    r_model, t_model = T_model(theta=theta)
    r_index = np.searchsorted(r_model, r)
    return -0.5 * np.sum((y - t_model[r_index]) ** 2 / yerr**2)

def log_prior(theta):
    a_max, Mstar, Mdot = theta
    if 0.001 < a_max < 10 and 0.08 < Mstar < 0.20 and 0.08 < Mdot < 0.20:
        return 0.0
    return -np.inf

def log_probability(theta, r, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, r, y, yerr)


# MCMC Sampler
def main(p0,nwalkers,niter,ndim,lnprob,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
    return sampler, pos, prob, state


def plotter(sampler, x=r_data, t=t_data, yerr=yerr):
    # plt.ion()
    plt.errorbar(x, t, yerr=yerr, fmt=".k", capsize=0,label='training data')
    samples = sampler.flatchain
    for theta in samples[np.random.randint(len(samples), size=100)]:
        r_model, t_model = T_model(theta)
        plt.plot(r_model, t_model, "C1", alpha=0.1)
    theta_max = samples[np.argmax(sampler.flatlnprobability)]
    print(theta_max)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend()
    plt.title('MCMC Test')
    plt.ylabel('$T_{mid}$ [K]')
    plt.xlabel('Radius [au]')
    plt.savefig('MCMC_test.pdf')
    # plt.show()

def posterior(sampler, label = ['$a_{max}$','$M_{*}$', '$\dot{M}$'], truth = [0.1, 0.14, 0.14]):
    samples = sampler.flatchain
    fig = corner.corner(
        samples, labels=label, truths=truth, 
        show_titles=True, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84]
        )
    fig.savefig('corner.pdf')

nwalkers, ndim = 50, 3  # Number of walkers and dimension of the parameter space
pos = [np.array([0.11, 0.13, 0.13]) + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]
sampler, pos, prob, state = main(pos,nwalkers,200,ndim,log_probability, (r_data, t_data, yerr))
plt.plot(r_grid, t_mid_fiducial, label = 'Fiducial Model')
plotter(sampler)
posterior(sampler)
