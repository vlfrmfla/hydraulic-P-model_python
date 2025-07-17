import math
from quad import quadm  # Assuming QUADM was implemented as quadm in a separate file
import numpy as np
from default_function.hyd_transpiration import calc_gsprime, calc_gs

def calc_assim_rubisco_limited(gs, vcmax, par_photosynth):
    ca = par_photosynth["ca"]
    gs_umol = gs * 1e6 / par_photosynth["patm"]
    d = par_photosynth["delta"]

    A = -1.0 * gs_umol
    B = gs_umol * ca - gs_umol * par_photosynth["kmm"] - vcmax * (1 - d)
    C = gs_umol * ca * par_photosynth["kmm"] + vcmax * (par_photosynth["gammastar"] + par_photosynth["kmm"] * d)

    ci = quadm(A, B, C)
    a_c = gs_umol * (ca - ci)

    return {"a": a_c, "ci": ci}

def calc_assim_light_limited(gs, jmax, par_photosynth):
    ca = par_photosynth["ca"]
    gs_umol = gs * 1e6 / par_photosynth["patm"]

    phi0iabs = par_photosynth["phi0"] * par_photosynth["Iabs"]
    jlim = phi0iabs / math.sqrt(1 + (4 * phi0iabs / jmax) ** 2)

    d = par_photosynth["delta"]
    A = -1.0 * gs_umol
    B = gs_umol * ca - gs_umol * 2 * par_photosynth["gammastar"] - jlim * (1 - d)
    C = gs_umol * ca * 2 * par_photosynth["gammastar"] + jlim * (par_photosynth["gammastar"] + d * par_photosynth["kmm"])

    ci = quadm(A, B, C)
    aj = gs_umol * (ca - ci)

    return {"a": aj, "ci": ci}

def calc_assimilation_limiting(vcmax, jmax, gs, par_photosynth):
    Ac = calc_assim_rubisco_limited(gs, vcmax, par_photosynth)
    Aj = calc_assim_light_limited(gs, jmax, par_photosynth)

    result = {"ac": Ac["a"], "aj": Aj["a"]}

    if Ac["ci"] > Aj["ci"]:
        result["ci"] = Ac["ci"]
        result["a"] = Ac["a"]
    else:
        result["ci"] = Aj["ci"]
        result["a"] = Aj["a"]

    return result


def calc_Aj_max(gs, x, par_photosynth):
    g = par_photosynth['gammastar'] / par_photosynth['ca']
    k = par_photosynth['kmm'] / par_photosynth['ca']
    ca = par_photosynth['ca'] / par_photosynth['patm'] * 1e6
    d = par_photosynth['delta']
    return gs * ca * (1 - x) * (x + 2 * g) / (x * (1 - d) - (g + d * k))

def calc_jmax_from_Ajmax(ajmax, par_photosynth):
    p = par_photosynth['phi0'] * par_photosynth['Iabs']
    return 4 * p / ( (p / ajmax) ** 2 - 1 ) ** 0.5

def calc_djmax_dAjmax(ajmax, par_photosynth):
    p = par_photosynth['phi0'] * par_photosynth['Iabs']
    return 4 * p**3 / ajmax**3 / ( (p / ajmax) ** 2 - 1 ) ** (3/2)

def calc_dAjmax_dchi(gs, x, par_photosynth):
    g = par_photosynth['gammastar'] / par_photosynth['ca']
    k = par_photosynth['kmm'] / par_photosynth['ca']
    ca = par_photosynth['ca'] / par_photosynth['patm'] * 1e6
    d = par_photosynth['delta']
    numerator = d*(2*g*(k+1) + k*(2*x-1) + x**2) - ((x-g)**2 + 3*g*(1-g))
    denominator = (d*(k+x) + g - x) ** 2
    return gs * ca * numerator / denominator

def calc_dAjmax_ddpsi(gsprime, x, par_photosynth):
    g = par_photosynth['gammastar'] / par_photosynth['ca']
    k = par_photosynth['kmm'] / par_photosynth['ca']
    ca = par_photosynth['ca'] / par_photosynth['patm'] * 1e6
    d = par_photosynth['delta']
    return gsprime * ca * (1 - x) * (x + 2 * g) / (x * (1 - d) - (g + d * k))

def calc_x_from_dpsi(dpsi, psi_soil, par_plant, par_env, par_photosynth, par_cost):
    gstar = par_photosynth['gammastar'] / par_photosynth['patm'] * 1e6
    Km = par_photosynth['kmm'] / par_photosynth['patm'] * 1e6
    ca = par_photosynth['ca'] / par_photosynth['patm'] * 1e6
    br = par_photosynth['delta']
    y = par_cost['gamma']

    gsprime = calc_gsprime(dpsi, psi_soil, par_plant, par_env)
    term1 = -2 * ca * dpsi * (gstar + br * Km) * y
    term2 = ca**2 * ((3 - 2*br) * gstar + br * Km) * gsprime
    sqrt_term = np.sqrt(2) * np.sqrt(
        ca**2 * dpsi * ((-3 + 2*br) * gstar - br * Km) * ((-1 + br) * ca + gstar + br * Km) * y *
        (-2 * dpsi * y + (ca + 2*gstar) * gsprime)
    )
    denominator = ca**2 * (2 * (-1 + br) * dpsi * y + ((3 - 2*br) * gstar + br * Km) * gsprime)
    x = (term1 + term2 - sqrt_term) / denominator
    # 최소값 제한 (벗어나는 경우 커트)
    min_x = (gstar + br * Km) / (ca - br * ca) + 1e-12
    if np.any(x < min_x):
        x = min_x
    return x

def chi_jmax_limited(par_photosynth, par_cost):
    g = par_photosynth['gammastar'] / par_photosynth['ca']
    k = par_photosynth['kmm'] / par_photosynth['ca']
    b = par_photosynth['delta']
    a = par_cost['alpha']
    # 아래는 원래 식에 그대로 대응 (복잡한 sqrt, 분모/분자)
    numerator = 2 * np.sqrt(
        -a * (4*a + b - 1) * (-3*g + 2*b*g - b*k) * (-1 + b + g + b*k)
    ) - (4*a + b - 1) * (b*k + g)
    denominator = (b - 1) * (4*a + b - 1)
    return numerator / denominator


def dFdx(dpsi, psi_soil, par_photosynth, par_plant, par_env, par_cost):
    gs = calc_gs(dpsi, psi_soil, par_plant, par_env)
    gsprime = calc_gsprime(dpsi, psi_soil, par_plant, par_env)
    X = calc_x_from_dpsi(dpsi, psi_soil, par_plant, par_env, par_photosynth, par_cost)
    ajmax = calc_Aj_max(gs, X, par_photosynth)
    ca = par_photosynth['ca'] / par_photosynth['patm'] * 1e6
    g = par_photosynth['gammastar'] / par_photosynth['ca']
    djmax_dajmax = calc_djmax_dAjmax(ajmax, par_photosynth)
    dajmax_dchi = calc_dAjmax_dchi(gs, X, par_photosynth)
    dP_dx = -gs * ca - par_cost['alpha'] * djmax_dajmax * dajmax_dchi
    return {
        "dP_dx": dP_dx,
        "ajmax": ajmax,
        "djmax_dajmax": djmax_dajmax,
        "dajmax_dchi": dajmax_dchi
    }