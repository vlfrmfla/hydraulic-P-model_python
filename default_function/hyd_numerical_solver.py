import numpy as np
from scipy.optimize import minimize
from .calc_gs import calc_gs
from .calc_assim_light_limited import calc_assim_light_limited
from .calc_vcmax_coordinated import calc_vcmax_coordinated_numerical  # 따로 구현해야 함

# Vcmax 조정 계산
def calc_vcmax_coordinated_numerical(aj, ci, par_photosynth):
    d = par_photosynth["delta"]
    kmm = par_photosynth["kmm"]
    gammastar = par_photosynth["gammastar"]
    numerator = aj * (ci + kmm)
    denominator = ci * (1 - d) - (gammastar + kmm * d)
    return numerator / denominator

# 수익 함수 (profit function)
def fn_profit(par, psi_soil, par_cost, par_photosynth, par_plant, par_env, do_optim=False, opt_hypothesis="PM"):
    jmax = np.exp(par[0])  # log(jmax) → jmax
    dpsi = par[1]

    gs = calc_gs(dpsi, psi_soil, par_plant, par_env)  # mol m-2 s-1 MPa-1
    E = 1.6 * gs * (par_env["vpd"] / par_env["patm"]) * 1e6  # umol m-2 s-1

    # 광 제한 동화율 Aj 계산
    a_j = calc_assim_light_limited(gs, jmax, par_photosynth)
    a = a_j["a"]
    ci = a_j["ci"]

    vcmax = calc_vcmax_coordinated_numerical(a, ci, par_photosynth)

    costs = par_cost["alpha"] * jmax + par_cost["gamma"] * dpsi**2
    benefit = 1  # optional: (1 + 1 / (par_photosynth["ca"] / 40.53)) / 2
    dummy_costs = 0 * np.exp(20 * (-abs(dpsi / 4) - abs(jmax / 1)))

    if opt_hypothesis == "PM":
        out = a * benefit - costs - dummy_costs
    elif opt_hypothesis == "LC":
        out = -(costs + dummy_costs) / (a + 1e-4)
    else:
        raise ValueError("opt_hypothesis must be 'PM' or 'LC'")

    return -out if do_optim else out

# 최적화 함수
def optimise_midterm_multi(fn_profit, psi_soil, par_cost, par_photosynth, par_plant, par_env, return_all=False, opt_hypothesis="PM"):
    bounds = [(-10, 10), (1e-4, 1e6)]
    x0 = [0, 1]  # initial guess: log(jmax), dpsi

    result = minimize(
        fn_profit,
        x0=x0,
        args=(psi_soil, par_cost, par_photosynth, par_plant, par_env, True, opt_hypothesis),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500}
    )

    # 음수 최소화를 반전해서 최대화 값 반환
    result.fun = -result.fun

    if return_all:
        return result
    else:
        return result.x
