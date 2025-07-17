# default_function/calc_ftemp_arrh.py

import numpy as np

def calc_ftemp_arrh(tk, dha, tkref=298.15):
    """
    Arrhenius-type 온도 반응 scaling factor 계산.
    (tk: K, dha: J/mol, tkref: K)
    
    Parameters
    ----------
    tk : float or array-like
        온도 (K)
    dha : float
        활성화 에너지 (J/mol)
    tkref : float, optional
        기준 온도 (K), 기본값 298.15 (25°C)
    
    Returns
    -------
    ftemp : float or ndarray
        온도 스케일링 팩터 (비율)
    """
    kR = 8.3145  # J/mol/K
    ftemp = np.exp(dha * (tk - tkref) / (tkref * kR * tk))
    return ftemp

# 예시:
# 25도(298.15K) → 10도(283.15K)에서 100,000 J/mol 활성화에너지로 온도효과(%감소):
# print((1.0 - calc_ftemp_arrh(283.15, 100000, 298.15)) * 100)
