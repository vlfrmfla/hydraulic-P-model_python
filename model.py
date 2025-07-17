# model.py
import numpy as np
from scipy.integrate import quad

ETA = 1.002e-3
CA = 400
PHI0 = 0.087
BR = 0.002
GAMMA_STAR = 42.75
KM = 404.9

class PlantModel:
    def __init__(self, Kp, psi50, b, alpha, gamma):
        self.Kp = Kp
        self.psi50 = psi50
        self.b = b
        self.alpha = alpha
        self.gamma = gamma

    def vulnerability_curve(self, psi):
        ratio = psi / self.psi50
        return 1 / (1 + np.abs(ratio) ** self.b)

    def water_flow(self, psi_s, delta_psi):
        psi_l = psi_s - delta_psi
        # 적분 구간 확인
        if psi_l >= psi_s:
            return 0.0
        integrand = lambda psi: self.vulnerability_curve(psi)
        integral, _ = quad(integrand, psi_l, psi_s)
        Q = self.Kp / ETA * integral  # 부호 확인
        return max(Q, 0.0)  # 음수 방지

    def stomatal_conductance(self, psi_s, delta_psi, D):
        E = self.water_flow(psi_s, delta_psi)
        gs = E / (1.6 * D)
        return max(gs, 1e-6)  # 완전히 0이 되지 않도록

    def electron_transport_capacity(self, Jmax, Iabs):
        numerator = 4 * PHI0 * Iabs
        denominator = np.sqrt(1 + (4 * PHI0 * Iabs / Jmax) ** 2)
        return numerator / denominator

    def Aj(self, chi, J, ca=CA, br=BR, gamma_star=GAMMA_STAR, km=KM):
        num = J * (chi * (1 - br) * ca - (gamma_star + br * km))
        denom = chi * ca + 2 * gamma_star
        return num / (4 * denom)

    def s17_equation(self, delta_psi, psi_s, Iabs, D, Jmax, ca=CA):
        # delta_psi 범위 체크
        if delta_psi <= 0:
            return np.inf
            
        gs = self.stomatal_conductance(psi_s, delta_psi, D)
        
        # chi 계산에서 분모가 0에 가까우면 문제
        denominator = ca * delta_psi
        if denominator < 1e-6:
            return np.inf
            
        chi = 1 - (1.6 * gs * D * self.gamma) / denominator
        
        # chi 범위 체크를 더 엄격하게
        if chi <= 0.01 or chi >= 0.99:
            return np.inf
            
        J = self.electron_transport_capacity(Jmax, Iabs)
        A = self.Aj(chi, J, ca)
        
        # A가 음수이면 문제
        if A <= 0:
            return np.inf
            
        lhs = ca * delta_psi * chi * (1 - chi)
        rhs = 1.6 * self.gamma * D * (self.gamma * delta_psi ** 2 + A)
        
        return lhs - rhs

    def safe_root_scalar(self, func, a, b, **kwargs):
        # 더 넓은 범위에서 해 찾기 시도
        ranges_to_try = [
            (a, b),
            (0.001, 5.0),
            (0.001, 10.0),
            (0.1, 20.0)
        ]
        
        for low, high in ranges_to_try:
            try:
                f_a = func(low)
                f_b = func(high)
                print(f"Trying bracket [{low:.3f}, {high:.3f}]: f(a)={f_a:.3e}, f(b)={f_b:.3e}")
                
                if not (np.isfinite(f_a) and np.isfinite(f_b)):
                    continue
                    
                if np.sign(f_a) != np.sign(f_b):
                    from scipy.optimize import root_scalar
                    return root_scalar(func, bracket=[low, high], **kwargs)
            except:
                continue
                
        raise RuntimeError("Root finding failed: no suitable bracket found")


    def semi_analytical_optimize(self, psi_s, Iabs, D, Jmax=150, ca=CA):
        def eq(delta_psi):
            return self.s17_equation(delta_psi, psi_s, Iabs, D, Jmax, ca)
        
        # 먼저 함수가 어떻게 동작하는지 체크
        test_points = np.linspace(0.01, 5.0, 50)
        valid_points = []
        for dp in test_points:
            val = eq(dp)
            if np.isfinite(val):
                valid_points.append((dp, val))
        
        if len(valid_points) < 2:
            raise RuntimeError("No valid points found for optimization")
        
        # 부호가 바뀌는 구간 찾기
        for i in range(len(valid_points)-1):
            dp1, val1 = valid_points[i]
            dp2, val2 = valid_points[i+1]
            if np.sign(val1) != np.sign(val2):
                sol = self.safe_root_scalar(eq, dp1, dp2, method='brentq')
                if sol.converged:
                    delta_psi_opt = sol.root
                    gs_opt = self.stomatal_conductance(psi_s, delta_psi_opt, D)
                    chi_opt = 1 - (1.6 * gs_opt * D * self.gamma) / (ca * delta_psi_opt)
                    A_opt = self.Aj(chi_opt, self.electron_transport_capacity(Jmax, Iabs), ca)
                    return {'chi': chi_opt, 'delta_psi': delta_psi_opt, 'gs': gs_opt, 'A': A_opt}
        
        raise RuntimeError("Semi-analytical optimization failed: no sign change found")
