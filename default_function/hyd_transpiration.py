import numpy as np
from scipy.special import gammainc
from .hyd_vulnerability_curve import P, Pprime
from scipy import integrate, special

# 줄기 도체율 계산 (m2)
def calc_conductivity_m(sapwood_perm, hv, height):
    return sapwood_perm * hv / height

# 단위 환산: 도체율을 mol/m2/s/MPa로 변환
def scale_conductivity(K, par_env):
    viscosity = par_env['viscosity_water']
    density = par_env['density_water']
    mol_h2o_per_kg = 55.5

    K2 = K / viscosity
    K3 = K2 * density * mol_h2o_per_kg
    K4 = K3 * 1e6
    return K4

# 해석적 취약성 곡선 적분
def integral_P_ana(dpsi, psi_soil, psi50, b):
    ps = psi_soil / psi50
    pl = (psi_soil - dpsi) / psi50
    l2 = np.log(2)
    def gammainc_R(a, x):
        return special.gammainc(a, x) * special.gamma(a)  # 순서 수정
    # 부호 수정
    return psi50 / b * l2 ** (-1 / b) * (gammainc_R(1 / b, l2 * pl ** b) - gammainc_R(1 / b, l2 * ps ** b))

# 취약성 곡선 적분 (수치)
def integral_P_num(dpsi, psi_soil, psi50, b):
    lower = psi_soil
    upper = psi_soil - dpsi
    result, _ = integrate.quad(lambda psi: P(psi, psi50, b), lower, upper)
    return result
    
# 취약성 곡선 적분 (근사)
def integral_P_approx(dpsi, psi_soil, psi50, b):
    return -dpsi * P(psi_soil - dpsi / 2, psi50, b)
  
# 잎기공 전도도(gs)
def calc_gs(dpsi, psi_soil, par_plant, par_env):
    K = scale_conductivity(par_plant['conductivity'], par_env)
    D = par_env['vpd'] / par_env['patm']
    return K / (1.6 * D) * -integral_P_ana(dpsi, psi_soil, par_plant['psi50'], par_plant['b'])

# dpsi에 대한 gs 미분 (해석)
def calc_gsprime_analytical(dpsi, psi_soil, par_plant, par_env):
    K = scale_conductivity(par_plant['conductivity'], par_env)
    D = par_env['vpd'] / par_env['patm']
    return K / (1.6 * D) * P(psi_soil - dpsi, par_plant['psi50'], par_plant['b'])

# dpsi에 대한 gs 미분 (근사)
def calc_gsprime_approx(dpsi, psi_soil, par_plant, par_env):
    K = scale_conductivity(par_plant['conductivity'], par_env)
    D = par_env['vpd'] / par_env['patm']
    psi = psi_soil - dpsi / 2
    return K / (1.6 * D) * (P(psi, par_plant['psi50'], par_plant['b']) - Pprime(psi, par_plant['psi50'], par_plant['b']) * dpsi / 2)

# 수치 미분 (검증용)
def calc_gsprime_numerical(dpsi, psi_soil, par_plant, par_env):
    return (calc_gs(dpsi + 0.01, psi_soil, par_plant, par_env) - calc_gs(dpsi, psi_soil, par_plant, par_env)) / 0.01


# 기본 선택
calc_gsprime = calc_gsprime_analytical
integral_P = integral_P_ana
