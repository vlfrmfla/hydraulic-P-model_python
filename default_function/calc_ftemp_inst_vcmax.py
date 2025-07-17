import numpy as np
import matplotlib.pyplot as plt

def calc_ftemp_arrh(tk, dha, tkref=298.15):
    """Arrhenius-type temperature response function"""
    R = 8.3145
    return np.exp(dha * (tk - tkref) / (tkref * R * tk))

def calc_ftemp_inst_vcmax(tcleaf, tcgrowth=None, tcref=25.0):
    """
    Kattge & Knorr (2007) 식에 따른 Vcmax의 온도 즉각 반응계수(비율)를 계산.
    Parameters
    ----------
    tcleaf : float or np.ndarray
        잎 온도(섭씨)
    tcgrowth : float, optional
        생장 온도(섭씨, 생략 시 잎 온도와 동일)
    tcref : float
        기준 온도(섭씨, 기본값 25도)
    Returns
    -------
    fv : float or np.ndarray
        온도 즉각 반응계수
    """
    # 상수
    Ha = 71513      # activation energy (J/mol)
    Hd = 200000     # deactivation energy (J/mol)
    Rgas = 8.3145   # gas constant (J/mol/K)
    a_ent = 668.39  # 엔트로피 오프셋 (J/mol/K)
    b_ent = 1.07    # 엔트로피 기울기 (J/mol/K^2)

    if tcgrowth is None:
        tcgrowth = tcleaf

    # Kelvin 변환
    tkref = tcref + 273.15
    tkleaf = np.array(tcleaf) + 273.15

    # 엔트로피 계산(섭씨로 넣는 것에 주의!)
    dent = a_ent - b_ent * tcgrowth

    # Arrhenius scaling
    fva = calc_ftemp_arrh(tkleaf, Ha, tkref)
    # 비활성화 보정
    fvb = (1 + np.exp((tkref * dent - Hd) / (Rgas * tkref))) / \
          (1 + np.exp((tkleaf * dent - Hd) / (Rgas * tkleaf)))
    fv = fva * fvb
    return fv
