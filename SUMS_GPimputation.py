print('===import')
from functools import partial
import os
import tempfile
os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp()
import arviz as az
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.scipy import optimize as jopt
import gc
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.diagnostics import summary
from tinygp import GaussianProcess, kernels
from astropy.table import Table, vstack
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import argparse
print('arg')
parser = argparse.ArgumentParser(
    prog='Imputer_from_start_to_end',
    description='Figure it out'
)

parser.add_argument('start', type=int, help='Start index (integer)')
parser.add_argument('end', type=int, help='End index (integer)')

args = parser.parse_args()

start = args.start
end = args.end
print('data')
#opening data and a file with the names of the right columns
atlas = Table.read("/home/r0937090/SUMS_imput/exEMU_ATLAS2.fits")
dt = Table.read("/home/r0937090/SUMS_imput/BOSSA_AB.fits")
wav=atlas['wavelength'].data
nam=atlas['name'].tolist()
enam=[]
for a in atlas['name']:
    enam.append('e_'+a)
inst = np.array(atlas['instrument'].data).astype(str)
print('code')
def SEDinit(dt, inter):
    mags = np.asarray(list(dt[nam][inter]), dtype=float)
    errs = np.asarray(list(dt[enam][inter]), dtype=float)
    wav_arr = np.asarray(wav, dtype=float)
    valid = ~np.isnan(mags)

    magvals = mags[valid]
    wavvals = wav_arr[valid]
    errmag = errs[valid]
    filter_idx = np.where(valid)[0]

    return np.asarray(wavvals, dtype=float), np.asarray(magvals, dtype=float), np.asarray(errmag), filter_idx

#Constants
X0 = 4.0 # Intercept of the mean linear model
NU = 2.0 # Degrees of freedom for the t-student
class Matern32Linear:
    names = ("sigma", "scale", "m0", "logT")
    @staticmethod
    def priors(ctx):
        return (dist.TruncatedNormal(loc=0.6, scale=0.5, low=0), dist.LogNormal(jnp.log(0.3), 0.5), dist.Normal(ctx["mu0"], 0.5), dist.Normal(ctx["logT"], 0.3))
    @staticmethod
    def build(p):
        return (kernels.quasisep.Matern32(sigma=p["sigma"], scale=p["scale"]), lambda v: p["m0"] + p["logT"] * (v - X0))

# GP setup
def sample_spec(spec, ctx, fixed):
    fixed = fixed or {}
    out = {}
    for n, d in zip(spec.names, spec.priors(ctx)):
        out[n] = jnp.asarray(fixed[n]) if n in fixed else numpyro.sample(n, d)
    return out

def gather_spec(spec, samples, idx, fixed, n_draw):
    fixed = fixed or {}
    return {n: (jnp.full(n_draw, fixed[n]) if n in fixed else jnp.asarray(samples[n])[idx]) for n in spec.names}

def mcmc_predict(mcmc, xg):
    c = mcmc._sed
    spec, x = c["spec"], c["x"]
    s = mcmc.get_samples()
    f = jnp.asarray(s["f"])
    n_tot = f.shape[0]
    p = gather_spec(spec, s, slice(None), c["fixed"], n_tot)
    xg = jnp.asarray(xg)
    @partial(jax.jit, static_argnames="spec")
    def batch(p, f, x, xg, *, spec):
        def one(p_i, f_i):
            k, mean_fn = spec.build(p_i)
            return GaussianProcess(k, x, diag=1e-6, mean=mean_fn).condition(f_i, xg).gp.loc
        return jax.vmap(one)(p, f)

    return np.asarray(batch(p, f, x, xg, spec=spec))

def sed_model(x, yerr, y=None, nu=3.0, fixed=None, ctx=None, spec=Matern32Linear):
    n = x.size
    p = sample_spec(spec, ctx, fixed)
    if nu is None:
        nu = numpyro.sample("nu", dist.Gamma(2.0, 0.2))
    k, mean_fn = spec.build(p)
    L = jnp.linalg.cholesky(k(x, x) + 1e-8 * jnp.eye(n))
    eta = numpyro.sample("eta", dist.Normal(0.0, 1.0).expand([n]).to_event(1))
    f = numpyro.deterministic("f", jax.vmap(mean_fn)(x) + L @ eta)
    numpyro.sample("obs", dist.StudentT(nu, f, yerr), obs=y)

def fit_mcmc(x, y, yerr, nu=3.0, fixed=None, spec=Matern32Linear, warmup=1000, samples=1000, chains=4, seed=0):
    ok = yerr < 1e3
    x, y, yerr = jnp.asarray(x[ok]), jnp.asarray(y[ok]), jnp.asarray(yerr[ok])
    idx_min, idx_max = jnp.argmin(x), jnp.argmax(x)
    dx = x[idx_max] - x[idx_min]
    logT = float((y[idx_max] - y[idx_min]) / dx) if dx > 0 else 0.0
    ctx = {"mu0": float(jnp.sum(y / yerr**2) / jnp.sum(1 / yerr**2)), "logT": logT,
           "x": x, "y": y, "yerr": yerr}
    model = partial(sed_model, ctx=ctx, spec=spec)
    kern = NUTS(model, init_strategy=init_to_median, target_accept_prob=0.9)
    mcmc = MCMC(kern, num_warmup=warmup, num_samples=samples, num_chains=chains, progress_bar=False, chain_method='parallel', jit_model_args=False)
    mcmc.run(jax.random.PRNGKey(seed), x, yerr, y=y, nu=nu, fixed=fixed, ctx=ctx, spec=spec)
    mcmc._sed = dict(spec=spec, fixed=fixed, nu=nu, x=x, y=y, yerr=yerr)
    return mcmc
# GP to imputed data
def MCMC_single(x,y,e, not_x):
    mcmc = fit_mcmc(x, y, e, spec=Matern32Linear, nu=NU, warmup=100, samples=500)
    samples=mcmc.get_samples()
    chains=mcmc.get_samples(group_by_chain=True)
    GPsummary=summary(chains)
    GPvar = {}

    for name in ["sigma", "scale", "m0", "logT"]:
        samps = samples[name]

        GPvar[name] = {
        'mean': np.mean(samps), "q05": np.quantile(samps, 0.05),
        "q25": np.quantile(samps, 0.25),
        "q50": np.quantile(samps, 0.50),
        "q75" : np.quantile(samps, 0.75),
        "q95": np.quantile(samps, 0.95),
        "rhat": GPsummary[name]["r_hat"]
        }

    x_all = np.sort(np.concatenate([x, not_x]))
    curves = mcmc_predict(mcmc, x_all) #impute x
    zimput=curves[:, np.isin(x_all, x)]
    zyim, e_zyim= np.median(zimput, axis=0), np.std(zimput, axis=0)
    z= np.abs(zyim-y)/np.sqrt(e_zyim**2)
    mask_z = z>2
    mask_z_good = z<2
    not_x = np.sort(np.concatenate([not_x, x[mask_z]]))
    mask_imputation=(not_x<np.max(x)) & (not_x>np.min(x))
    imput=curves[:, np.isin(x_all, not_x[mask_imputation])]
    yim, e_yim= np.median(imput, axis=0), np.std(imput, axis=0)
    #construct returnable sets
    x_removed=10**np.asarray(x)[mask_z]
    x_imputed=10**np.asarray(not_x[mask_imputation])
    return yim, e_yim, x_removed, x_imputed, GPvar
# full dataset handling
def wav_to_filter(x):
    x = np.asarray(x, dtype=float)
    out = []
    for lam in x:
        a = np.argmin(np.abs(wav - lam))
        out.append(nam[a])
    return np.asarray(out)
def Master(start, end):
    filled = dt[start:end].copy()
    for name in ["sigma", "scale", "m0", "logT"]:
            for extra in ['q05', 'q25', 'q50', 'q75', 'q95', 'mean', 'rhat']:
                filled[f"{name}_{extra}"] = np.nan
    for a in nam:
        filled[f"{a}_impflag"] = 0.0

    for k in range(start, end):
        i = k - start
        print(i)
        x, y, e, fidx = SEDinit(filled, i)
        not_x = [wave for wave in wav if wave not in x]
        x = np.log10(x)
        not_x = np.log10(np.array(not_x))
        yim, e_yim, xrem, ximp, GPvar = MCMC_single(x, y, e, not_x)

        for name in ["sigma", "scale", "m0", "logT"]:
            for extra in ['q05', 'q25', 'q50', 'q75', 'q95', 'mean', 'rhat']:
                filled[f"{name}_{extra}"][i] = GPvar[name][extra]
        ab, xrem, ximp = (wav_to_filter(10**x),wav_to_filter(xrem),wav_to_filter(ximp))
        for band in xrem:
            filled[f"{band}_impflag"][i]=filled[band][i]
            filled[band][i]=np.nan
        for j, band in enumerate(ximp):
            filled[band][i]=yim[j]
            errimp="e_"+band
            filled[errimp][i]=e_yim[j]
            if filled[f"{band}_impflag"][i]==0.0:
                filled[f"{band}_impflag"][i]=-1
        del x, y, e, not_x, yim, e_yim, xrem, ximp, GPvar
        gc.collect()
        jax.clear_caches()
    return filled
filled=Master(start, end)
filled.write(f"/home/r0937090/SUMS_imput/filled/filled{start}_{end}.fits", format="fits", overwrite=True)




