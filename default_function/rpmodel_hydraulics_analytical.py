import numpy as np
from scipy.optimize import root_scalar
from scipy.special import gammainc, gamma
from quad import quadm
from default_function.hyd_vulnerability_curve import P, Pprime, Pprimeprime
from default_function.hyd_transpiration import calc_gs, calc_gsprime, scale_conductivity
from default_function.hyd_photosynthesis import calc_jmax_from_Ajmax, chi_jmax_limited, calc_Aj_max, calc_dAjmax_dchi, calc_dAjmax_ddpsi, calc_djmax_dAjmax, calc_x_from_dpsi, dFdx



def calc_dpsi_bound(psi_soil, par_plant, par_env, par_photosynth, par_cost):
    gstar = par_photosynth['gammastar'] / par_photosynth['patm'] * 1e6
    ca = par_photosynth['ca'] / par_photosynth['patm'] * 1e6
    y = par_cost['gamma']
    K = scale_conductivity(par_plant['conductivity'], par_env)
    K = K / (1.6 * par_env['vpd'] / par_env['patm'])
    Pox = P(psi_soil, par_plant['psi50'], par_plant['b'])
    Ppox = Pprime(psi_soil, par_plant['psi50'], par_plant['b'])
    Pppox = Pprimeprime(psi_soil, par_plant['psi50'], par_plant['b'])

    a = (ca + 2*gstar) * K * Pppox * 4 / 8
    b = -(2*y + (ca + 2*gstar) * K * Ppox)
    c = (ca + 2*gstar) * K * Pox
    disc = b**2 - 4*a*c
    approx_O2 = (-b - np.sqrt(disc)) / (2 * a) if disc >= 0 else np.nan

    def eq_dpsi(dpsi):
        return -2*dpsi*y + (ca + 2*gstar) * calc_gsprime(dpsi, psi_soil, par_plant, par_env)
    sol = root_scalar(eq_dpsi, bracket=[0, 10], method="brentq")
    exact = sol.root if sol.converged else np.nan

    def f1(dpsi):
        gs = calc_gs(dpsi, psi_soil, par_plant, par_env)
        x = calc_x_from_dpsi(dpsi, psi_soil, par_plant, par_env, par_photosynth, par_cost)
        ajmax = calc_Aj_max(gs, x, par_photosynth) - par_photosynth['phi0'] * par_photosynth['Iabs']
        return ajmax

    Iabs_bound = root_scalar(f1, bracket=[exact*0.001, exact*0.99], method="brentq").root

    return {"exact": exact, "Iabs_bound": Iabs_bound, "approx_O2": approx_O2}


def rpmodel_hydraulics_analytical(tc, ppfd, vpd, co2, elv, fapar, kphio, psi_soil, par_plant, par_photosynth, par_cost=None):
    p = par_photosynth['patm']
    par_photosynth_now = par_photosynth.copy()
    par_photosynth_now['phi0'] = kphio * par_photosynth['ftemp_kphio']
    par_photosynth_now['Iabs'] = ppfd * fapar
    par_photosynth_now['ca'] = co2 * p * 1e-6

    par_env_now = {
        "viscosity_water": par_photosynth['viscosity_water'],
        "density_water": par_photosynth['density_water'],
        "patm": p,
        "tc": tc,
        "vpd": vpd
    }

    if par_cost is None:
        par_cost_now = {"alpha": 0.1, "gamma": 1}
    else:
        par_cost_now = par_cost

    dpsi_bounds = calc_dpsi_bound(psi_soil, par_plant, par_env_now, par_photosynth_now, par_cost_now)
    dpsi_max = dpsi_bounds["Iabs_bound"]

    u = root_scalar(
        lambda dpsi: dFdx(dpsi, psi_soil, par_photosynth_now, par_plant, par_env_now, par_cost_now)["dP_dx"],
        bracket=[dpsi_max * 0.001, dpsi_max * 0.99],
        method="brentq"
    )
    dpsi = u.root
    x = calc_x_from_dpsi(dpsi, psi_soil, par_plant, par_env_now, par_photosynth_now, par_cost_now)
    gs = calc_gs(dpsi, psi_soil, par_plant, par_env_now)
    J = calc_Aj_max(gs, x, par_photosynth_now)
    Jmax = calc_jmax_from_Ajmax(J, par_photosynth_now)

    ca = par_photosynth_now["ca"] / par_photosynth_now["patm"] * 1e6
    kmm = par_photosynth_now["kmm"] / par_photosynth_now["patm"] * 1e6
    g = par_photosynth_now["gammastar"] / par_photosynth_now["patm"] * 1e6
    Vcmax = (J / 4) * (x * ca + kmm) / (x * ca + 2 * g)
    A = gs * ca * (1 - x)

    return {
        "jmax": Jmax,
        "dpsi": dpsi,
        "gs": gs,
        "a": A,
        "ci": x * par_photosynth_now["ca"],
        "chi": x,
        "chi_jmax_lim": chi_jmax_limited(par_photosynth_now, par_cost_now),
        "vcmax": Vcmax,
        "profit": A - par_cost_now["alpha"] * Jmax - par_cost_now["gamma"] * dpsi**2,
        "niter": 0,
        "nfcnt": 0,
        "dpsi_max_exact": dpsi_bounds["exact"],
        "dpsi_max_approx": dpsi_bounds["approx_O2"],
        "dpsi_max_Iabs_bound": dpsi_bounds["Iabs_bound"]
    }
