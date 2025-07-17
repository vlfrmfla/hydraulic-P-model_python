# default_function/calc_density_h2o.py

def calc_density_h2o(tc, p):
    """
    Calculates the density of water (kg/m^3) as a function of temperature (deg C) and pressure (Pa)
    using the Tumlirz Equation.

    Parameters
    ----------
    tc : float
        Air temperature (deg C)
    p : float
        Atmospheric pressure (Pa)

    Returns
    -------
    rho : float
        Density of water (kg/m^3)
    """
    # lambda (bar cm^3/g)
    my_lambda = (
        1788.316 +
        21.55053 * tc +
        -0.4695911 * tc**2 +
        (3.096363e-3) * tc**3 +
        -(7.341182e-6) * tc**4
    )

    # po (bar)
    po = (
        5918.499 +
        58.05267 * tc +
        -1.1253317 * tc**2 +
        (6.6123869e-3) * tc**3 +
        -(1.4661625e-5) * tc**4
    )

    # vinf (cm^3/g)
    vinf = (
        0.6980547 +
        -(7.435626e-4) * tc +
        (3.704258e-5) * tc**2 +
        -(6.315724e-7) * tc**3 +
        (9.829576e-9) * tc**4 +
        -(1.197269e-10) * tc**5 +
        (1.005461e-12) * tc**6 +
        -(5.437898e-15) * tc**7 +
        (1.69946e-17) * tc**8 +
        -(2.295063e-20) * tc**9
    )

    # Pressure in bar
    pbar = 1e-5 * p

    # Specific volume (cm^3/g)
    v = vinf + my_lambda / (po + pbar)

    # Density (kg/m^3): (1 g/cm^3 = 1000 kg/m^3)
    rho = 1e3 / v

    return rho

# 예시:
# print(calc_density_h2o(20, 101325))  # 20도C, 표준기압에서 약 998.2 kg/m3
