def calc_soilmstress(
    soilm,
    meanalpha=1.0,
    apar_soilm=0.0,
    bpar_soilm=0.685
):
    """
    토양수분 스트레스 팩터(beta) 계산 (Stocker et al., 2019)
    Parameters
    ----------
    soilm : float
        토양수분 (field capacity 대비 비율, 0~1)
    meanalpha : float, optional
        지역 연평균 실제/잠재 증발산 비율(건조도 지표), 기본 1.0
    apar_soilm : float, optional
        민감도 계수 a, 기본 0.0
    bpar_soilm : float, optional
        민감도 계수 b, 기본 0.685
    Returns
    -------
    beta : float
        토양수분 스트레스 인자 (0~1)
    Reference
    ---------
    Stocker et al., Geosci. Model Dev., 2019
    """
    x0 = 0.0
    x1 = 0.6  # threshold

    if soilm > x1:
        outstress = 1.0
    else:
        y0 = apar_soilm + bpar_soilm * meanalpha
        beta = (1.0 - y0) / (x0 - x1) ** 2
        outstress = 1.0 - beta * (soilm - x1) ** 2
        # 범위 강제(0~1)
        outstress = max(0.0, min(1.0, outstress))

    return outstress
