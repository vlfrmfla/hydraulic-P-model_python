# default_function/calc_patm.py

def calc_patm(elv, patm0=101325):
    """
    Calculates atmospheric pressure as a function of elevation.

    Parameters
    ----------
    elv : float or array-like
        Elevation above sea level (m.a.s.l.)
    patm0 : float, optional
        Atmospheric pressure at sea level (Pa). Default is 101325 Pa.

    Returns
    -------
    patm : float or array-like
        Atmospheric pressure at elevation 'elv' (Pa)
    """
    # Constants
    kTo = 298.15     # Standard temperature, K
    kL  = 0.0065     # Adiabatic lapse rate, K/m
    kG  = 9.80665    # Gravitational acceleration, m/s^2
    kR  = 8.3145     # Universal gas constant, J/mol/K
    kMa = 0.028963   # Molecular weight of dry air, kg/mol

    patm = patm0 * (1.0 - kL * elv / kTo) ** (kG * kMa / (kR * kL))
    return patm

# 예시:
# print(calc_patm(1000))
