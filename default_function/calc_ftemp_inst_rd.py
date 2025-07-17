# default_function/calc_ftemp_inst_rd.py

import numpy as np

def calc_ftemp_inst_rd(tc):
    """
    암호흡(암호화) 온도 스케일링 함수 (Heskel et al. 2016)
    ---------------------------------------------------------
    25°C에서의 암호흡 기준으로, 입력 온도(tc, °C)에서 암호흡의 scaling factor를 반환.
    공식:
        fr = exp( 0.1012*(T - 25) - 0.0005*(T^2 - 25^2) )
    참고: T(온도)는 섭씨(degree Celsius) 단위여야 함.

    Parameters
    ----------
    tc : float or array-like
        온도 (degrees Celsius)

    Returns
    -------
    fr : float or ndarray
        암호흡 온도 반응 scaling factor (25도에서 1, 낮을수록 작음)

    Examples
    --------
    >>> calc_ftemp_inst_rd(25)
    1.0
    >>> calc_ftemp_inst_rd(10)
    0.2529...

    Reference
    ---------
    Heskel, M. A., et al. (2016). 
    "Convergence in the temperature response of leaf respiration across biomes and plant functional types."
    Proceedings of the National Academy of Sciences, 113(14), 3832–3837. https://doi.org/10.1073/pnas.1520282113
    """
    apar = 0.1012
    bpar = 0.0005
    fr = np.exp(apar * (tc - 25.0) - bpar * (tc**2 - 25.0**2))
    return fr

# 사용 예시
if __name__ == "__main__":
    print("25도C:", calc_ftemp_inst_rd(25))     # 1.0
    print("10도C:", calc_ftemp_inst_rd(10))     # 약 0.253
    print("상대 변화율 (10→25도):", (calc_ftemp_inst_rd(25)/calc_ftemp_inst_rd(10)-1)*100, "%")
