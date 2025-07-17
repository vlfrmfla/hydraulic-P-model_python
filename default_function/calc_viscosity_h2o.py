import numpy as np

def calc_density_h2o(tc, p):
    """
    주어진 온도(℃)와 압력(Pa)에서 물의 밀도(kg/m^3) 계산.
    - tc : 온도 (섭씨)
    - p : 압력 (Pa)
    """
    my_lambda = 1788.316 + 21.55053*tc - 0.4695911*tc**2 + 3.096363e-3*tc**3 - 7.341182e-6*tc**4
    po = 5918.499 + 58.05267*tc - 1.1253317*tc**2 + 6.6123869e-3*tc**3 - 1.4661625e-5*tc**4
    vinf = 0.6980547 - 7.435626e-4*tc + 3.704258e-5*tc**2 - 6.315724e-7*tc**3 \
           + 9.829576e-9*tc**4 - 1.197269e-10*tc**5 + 1.005461e-12*tc**6 \
           - 5.437898e-15*tc**7 + 1.69946e-17*tc**8 - 2.295063e-20*tc**9
    pbar = 1e-5 * p  # Pa to bar
    v = vinf + my_lambda / (po + pbar)  # cm3/g
    rho = 1e3 / v  # kg/m3
    return rho

def calc_viscosity_h2o(tc, p):
    """
    물의 점도(Pa s) 계산 (Huber et al., 2009)
    Parameters
    ----------
    tc : float
        온도(℃)
    p : float
        대기압(Pa)
    Returns
    -------
    mu : float
        물의 점도(Pa s)
    """
    # 상수
    tk_ast  = 647.096    # Kelvin
    rho_ast = 322.0      # kg/m^3
    mu_ast  = 1e-6       # Pa s

    rho = calc_density_h2o(tc, p)
    tbar  = (tc + 273.15) / tk_ast
    tbarx = tbar ** 0.5
    tbar2 = tbar ** 2
    tbar3 = tbar ** 3
    rbar  = rho / rho_ast

    # mu0 (기본 점도)
    mu0 = 1.67752 + 2.20462/tbar + 0.6366564/tbar2 - 0.241605/tbar3
    mu0 = 1e2 * tbarx / mu0

    # Huber et al. (2009) Table 3 계수
    h_array = np.zeros((7, 6))
    h_array[0,:] = [0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0]
    h_array[1,:] = [0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573]
    h_array[2,:] = [-0.281378, -0.906851, -0.772479, -0.489837, -0.257040, 0.0]
    h_array[3,:] = [0.161913, 0.257399, 0.0, 0.0, 0.0, 0.0]
    h_array[4,:] = [-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0]
    h_array[5,:] = [0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0]
    h_array[6,:] = [0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264]

    # mu1 계산
    mu1 = 0.0
    ctbar = (1.0 / tbar) - 1.0
    for i in range(6):
        coef1 = ctbar ** i
        coef2 = 0.0
        for j in range(7):
            coef2 += h_array[j, i] * (rbar - 1.0) ** j
        mu1 += coef1 * coef2
    mu1 = np.exp(rbar * mu1)

    mu_bar = mu0 * mu1  # 단위없는 점도
    mu = mu_bar * mu_ast  # Pa s

    return mu
