# default_function/calc_gammastar.py

from .calc_ftemp_arrh import calc_ftemp_arrh
from .calc_patm import calc_patm

def calc_gammastar(tc, patm):
    """
    광호흡 CO₂ 보상점 (gammastar, Pa) 계산

    -----------------------------------------------------------------------
    주어진 온도(섭씨)와 대기압(Pa)에서의 Rubisco 광합성 모델의
    광호흡 CO₂ 보상점(gammastar, Pa)을 계산합니다.
    (Farquhar et al., 1980; Bernacchi et al., 2001)
    
    Parameters
    ----------
    tc : float
        광합성에 적용되는 온도 (섭씨)
    patm : float
        대기압 (Pa)

    Returns
    -------
    gammastar : float
        온도 및 대기압 보정된 광호흡 CO₂ 보상점 (Pa)

    Details
    -------
    - 표준(25°C, 해수면 대기압) 보상점: 4.332 Pa
    - 활성화 에너지(Arrhenius): 37830 J/mol
    - 보정 공식:
        gammastar = gs25_0 * patm / calc_patm(0) * calc_ftemp_arrh(tc + 273.15, dha)

    References
    ----------
    Farquhar, G. D., von Caemmerer, S., & Berry, J. A. (1980).
        A biochemical model of photosynthetic CO2 assimilation in leaves of C3 species.
        Planta, 149, 78–90.
    Bernacchi, C. J., et al. (2001).
        Improved temperature response functions for models of Rubisco-limited photosynthesis.
        Plant, Cell and Environment, 24, 253–259.

    Examples
    --------
    >>> calc_gammastar(20, 101325)
    # 20도C, 해수면 대기압에서의 gammastar (Pa)
    """
    dha = 37830         # J/mol, Bernacchi et al. (2001)
    gs25_0 = 4.332      # Pa, Bernacchi et al. (2001)
    gammastar = gs25_0 * patm / calc_patm(0.0) * calc_ftemp_arrh(tc + 273.15, dha=dha)
    return gammastar

# 예시:
if __name__ == "__main__":
    print("CO2 compensation point at 20C, 101325Pa:", calc_gammastar(20, 101325))
