import numpy as np
import matplotlib.pyplot as plt
import disk_model
import emcee
import corner

def T_model(theta):
    a_max, Mstar = theta
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
        Mstar=Mstar*disk_model.Msun, Mdot=Mstar*1e-5*disk_model.Msun/disk_model.yr,
        Rd=30*disk_model.au,Q=1.5,N_R=50
        )
    R_grid = DM.R[1:]/disk_model.au
    T_mid = DM.T_mid
    return R_grid, T_mid

r_grid, t_mid_fiducial = T_model(theta = (0.1, 0.14))
yerr = 20
t_mid_train = t_mid_fiducial + np.random.normal(loc=0.0, scale=yerr, size=len(t_mid_fiducial))
# plt.plot(r_grid, t_mid_train)
# plt.show()

def log_likelihood(theta, y, yerr):
    _, t_model = T_model(theta=theta)
    sigma2 = yerr**2
    return -0.5 * np.sum((y - t_model) ** 2 / sigma2)

def log_prior(theta):
    a_max, Mstar = theta
    if 0.001 < a_max < 1 and 0.08 < Mstar < 0.20:
        return 0.0
    return -np.inf

def log_probability(theta, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, yerr)

def main(p0,nwalkers,niter,ndim,lnprob,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
    pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
    return sampler, pos, prob, state


def plotter(sampler, x=r_grid, t=t_mid_train):
    # plt.ion()
    plt.plot(x, t, label='training data')
    samples = sampler.flatchain
    for theta in samples[np.random.randint(len(samples), size=200)]:
        _, t_model = T_model(theta)
        plt.plot(x, t_model, color='gray', alpha=0.1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend()
    plt.show()


# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(t_mid_train, yerr))
# sampler.run_mcmc(pos, 5, progress = True)
nwalkers, ndim = 32, 2  # Number of walkers and dimension of the parameter space
pos = [np.array([0.09, 0.13]) + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
sampler, pos, prob, state = main(pos,nwalkers,500,ndim,log_probability, (t_mid_train, yerr))
plotter(sampler)
# 