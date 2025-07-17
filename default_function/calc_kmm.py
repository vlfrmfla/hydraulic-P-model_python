# default_function/calc_kmm.py

import numpy as np
from .calc_ftemp_arrh import calc_ftemp_arrh

def calc_kmm(tc, patm):
    """
    Rubisco 한계 광합성을 위한 Michaelis-Menten 계수 계산
    ----------------------------------------------------------------------
    Farquhar et al. (1980) 및 Bernacchi et al. (2001)에 기반하여,
    온도(섭씨)와 대기압(Pa)을 받아 Michaelis-Menten 계수(K, Pa) 반환.

    Parameters
    ----------
    tc : float
        광합성 온도 (degrees Celsius)
    patm : float
        대기압 (Pa)

    Returns
    -------
    kmm : float
        Michaelis-Menten coefficient (Pa)

    Details
    -------
    Rubisco 한계 광합성의 Michaelis-Menten 계수 K는 아래와 같이 계산됨:
        K = Kc * (1 + pO2 / Ko)
    - Kc, Ko는 각각 CO2, O2에 대한 Michaelis-Menten 계수 (Pa)
    - pO2 = 0.209476 * patm

    Kc, Ko의 온도 반응은 Arrhenius 함수(calc_ftemp_arrh)로 보정.
    (Bernacchi et al. 2001: Kc25=39.97 Pa, Ko25=27480 Pa, 
    dhac=79430 J/mol, dhao=36380 J/mol)

    Examples
    --------
    >>> calc_kmm(20, 101325)
    # (20도C, 해수면 대기압에서 Michaelis-Menten 계수(Pa) 반환)

    References
    ----------
    Farquhar, G. D., von Caemmerer, S., & Berry, J. A. (1980).
        A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species.
        Planta, 149, 78–90.
    Bernacchi, C. J., et al. (2001).
        Improved temperature response functions for models of Rubisco-limited photosynthesis.
        Plant, Cell and Environment, 24, 253–259.
    """
    dhac = 79430      # J/mol, Bernacchi et al. 2001
    dhao = 36380      # J/mol, Bernacchi et al. 2001
    kco = 2.09476e5   # ppm, O2 partial pressure in air

    kc25 = 39.97      # Pa, Bernacchi et al. 2001
    ko25 = 27480      # Pa, Bernacchi et al. 2001

    tk = tc + 273.15

    kc = kc25 * calc_ftemp_arrh(tk, dha=dhac)
    ko = ko25 * calc_ftemp_arrh(tk, dha=dhao)

    po = kco * (1e-6) * patm  # O2 partial pressure (Pa)
    kmm = kc * (1.0 + po / ko)

    return kmm

# 예시:
if __name__ == "__main__":
    print("Michaelis-Menten coefficient at 20C, 101325Pa:", calc_kmm(20, 101325))
