# default_function/calc_ftemp_kphio.py

def calc_ftemp_kphio(tc):
    """
    광양자 수율 효율(quantum yield efficiency)의 온도 의존성 계산

    -----------------------------------------------------------------------
    광합성 온도(섭씨)에서의 최대 광양자 수율 효율(photosystem II의 온도 의존성)을
    Bernacchi et al. (2003) 의 식에 따라 계산합니다.

    Parameters
    ----------
    tc : float
        광합성 관련 온도(섭씨)

    Returns
    -------
    ftemp : float
        온도 보정된 quantum yield efficiency

    Details
    -------
    - 온도 의존 공식:
        φ(T) = 0.352 + 0.022 * T - 0.00034 * T^2
      여기서 T는 섭씨 온도입니다.

    References
    ----------
    Bernacchi, C. J., Pimentel, C., & Long, S. P. (2003).
        In vivo temperature response functions of parameters required to model RuBP-limited photosynthesis.
        Plant, Cell & Environment, 26, 1419–1430.

    Examples
    --------
    >>> # 25도C에서의 값
    >>> calc_ftemp_kphio(25.0)
    >>> # 5도C와 25도C 변화율
    >>> (calc_ftemp_kphio(25.0)/calc_ftemp_kphio(5.0)-1)*100
    """
    ftemp = 0.352 + 0.022 * tc - 3.4e-4 * tc ** 2
    return ftemp

# 예시
if __name__ == "__main__":
    print("25°C:", calc_ftemp_kphio(25.0))
    print("5°C :", calc_ftemp_kphio(5.0))
    print("Percent change 5→25°C: {:.2f}%".format((calc_ftemp_kphio(25.0)/calc_ftemp_kphio(5.0)-1)*100))
