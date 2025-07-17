# rpmodel.py
import numpy as np
from scipy.optimize import minimize, root_scalar
import warnings

# Import required functions (assuming they exist in default_function)
from default_function.calc_patm import calc_patm
from default_function.calc_gammastar import calc_gammastar
from default_function.calc_kmm import calc_kmm
from default_function.calc_viscosity_h2o import calc_viscosity_h2o
from default_function.calc_ftemp_kphio import calc_ftemp_kphio
from default_function.calc_soilmstress import calc_soilmstress
from default_function.calc_ftemp_inst_vcmax import calc_ftemp_inst_vcmax
from default_function.calc_ftemp_inst_rd import calc_ftemp_inst_rd
from default_function.quad import quadm

def co2_to_ca(co2, patm):
    """Convert CO2 from ppm to Pa"""
    ca = (1.0e-6) * co2 * patm
    return ca

def calc_ftemp_inst_jmax(tc, tcgrowth=None, tcref=25.0):
    """Temperature response for Jmax (simplified version)"""
    if tcgrowth is None:
        tcgrowth = tc
    return calc_ftemp_inst_vcmax(tc, tcgrowth, tcref)

def calc_optimal_chi(kmm, gammastar, ns_star, ca, vpd, beta):
    """Calculate optimal chi following Prentice et al. 2014"""
    # Avoid negative VPD
    vpd = max(vpd, 0)
    
    # Optimal ci:ca ratio
    xi = np.sqrt((beta * (kmm + gammastar)) / (1.6 * ns_star))
    chi = gammastar / ca + (1.0 - gammastar / ca) * xi / (xi + np.sqrt(vpd))
    
    # Calculate factors
    vdcg = ca - gammastar
    vacg = ca + 2.0 * gammastar
    vbkg = beta * (kmm + gammastar)
    
    def calc_mj_single(ns_star, vpd, vbkg):
        if ns_star > 0 and vpd > 0 and vbkg > 0:
            vsr = np.sqrt(1.6 * ns_star * vpd / vbkg)
            mj = vdcg / (vacg + 3.0 * gammastar * vsr)
        else:
            mj = np.nan
        return mj
    
    mj = calc_mj_single(ns_star, vpd, vbkg)
    
    gamma = gammastar / ca
    kappa = kmm / ca
    
    mc = (chi - gamma) / (chi + kappa)
    mjoc = (chi + kappa) / (chi + 2 * gamma)
    
    return {'chi': chi, 'mc': mc, 'mj': mj, 'mjoc': mjoc}

def calc_chi_c4():
    """Dummy chi for C4 photosynthesis"""
    return {'chi': 9999, 'mc': 1, 'mj': 1, 'mjoc': 1}

def calc_mprime(mc, kc):
    """Calculate modified m accounting for co-limitation"""
    mpi = mc**2 - kc**(2.0/3.0) * (mc**(4.0/3.0))
    return np.sqrt(mpi) if mpi > 0 else np.nan

def calc_lue_vcmax_wang17(out_optchi, kphio, ftemp_kphio, c_molmass, soilmstress, c_cost):
    """Calculate LUE and Vcmax using Wang17 method"""
    mprime = calc_mprime(out_optchi['mj'], c_cost)
    
    return {
        'lue': kphio * ftemp_kphio * mprime * c_molmass * soilmstress,
        'vcmax_unitiabs': kphio * ftemp_kphio * out_optchi['mjoc'] * mprime / out_optchi['mj'] * soilmstress,
        'omega': np.nan,
        'omega_star': np.nan
    }

def calc_lue_vcmax_smith19(out_optchi, kphio, ftemp_kphio, c_molmass, soilmstress):
    """Calculate LUE and Vcmax using Smith19 method"""
    def calc_omega(theta, c_cost, m):
        cm = 4 * c_cost / m
        v = 1/(cm * (1 - theta * cm)) - 4 * theta
        
        # Non-linearities at low m values
        capP = (((1/1.4) - 0.7)**2 / (1-theta)) + 3.4
        coeffs = [-1, capP, -(capP * theta)]
        roots = np.roots(coeffs)
        m_star = (4 * c_cost) / roots[0].real
        
        if m < m_star:
            omega = -(1 - (2 * theta)) - np.sqrt((1 - theta) * v)
        else:
            omega = -(1 - (2 * theta)) + np.sqrt((1 - theta) * v)
        return omega
    
    theta = 0.85
    c_cost = 0.05336251
    
    omega = calc_omega(theta, c_cost, out_optchi['mj'])
    omega_star = 1.0 + omega - np.sqrt((1.0 + omega)**2 - (4.0 * theta * omega))
    
    mprime = out_optchi['mj'] * omega_star / (8.0 * theta)
    lue = kphio * ftemp_kphio * mprime * c_molmass * soilmstress
    vcmax_unitiabs = kphio * ftemp_kphio * out_optchi['mjoc'] * omega_star / (8.0 * theta) * soilmstress
    
    return {
        'lue': lue,
        'vcmax_unitiabs': vcmax_unitiabs,
        'omega': omega,
        'omega_star': omega_star
    }

def calc_lue_vcmax_none(out_optchi, kphio, ftemp_kphio, c_molmass, soilmstress):
    """Calculate LUE and Vcmax without Jmax limitation"""
    return {
        'lue': kphio * ftemp_kphio * out_optchi['mj'] * c_molmass * soilmstress,
        'vcmax_unitiabs': kphio * ftemp_kphio * out_optchi['mjoc'] * soilmstress,
        'omega': np.nan,
        'omega_star': np.nan
    }

def calc_lue_vcmax_c4(kphio, ftemp_kphio, c_molmass, soilmstress):
    """Calculate LUE and Vcmax for C4 plants"""
    return {
        'lue': kphio * ftemp_kphio * c_molmass * soilmstress,
        'vcmax_unitiabs': kphio * ftemp_kphio * soilmstress,
        'omega': np.nan,
        'omega_star': np.nan
    }

def calc_optim_num(kmm, gammastar, ns_star, ca, vpd, ppfd, fapar, kphio, beta, c_cost, 
                   vcmax_start, gs_start, jmax_start):
    """Numerical optimization of gs, vcmax, and jmax"""
    
    def optimise_this_gs_vcmax_jmax(par, args, iabs, kphio, beta, c_cost, maximize=False, return_all=False):
        kmm, gammastar, ns_star, ca, vpd = args
        vcmax, gs, jmax = par
        
        # Electron transport limiting
        L = 1.0 / np.sqrt(1.0 + ((4.0 * kphio * iabs)/jmax)**2)
        A = -gs
        B = gs * ca - 2 * gammastar * gs - L * kphio * iabs
        C = 2 * gammastar * gs * ca + L * kphio * iabs * gammastar
        
        ci_j = quadm(A, B, C)
        a_j = kphio * iabs * (ci_j - gammastar)/(ci_j + 2 * gammastar) * L
        
        # Rubisco limiting
        A = -1.0 * gs
        B = gs * ca - gs * kmm - vcmax
        C = gs * ca * kmm + vcmax * gammastar
        
        ci_c = quadm(A, B, C)
        a_c = vcmax * (ci_c - gammastar) / (ci_c + kmm)
        
        # Take minimum assimilation and maximum ci
        assim = min(a_j, a_c)
        ci = max(ci_c, ci_j)
        
        # Costs
        cost_transp = 1.6 * ns_star * gs * vpd
        cost_vcmax = beta * vcmax
        cost_jmax = c_cost * jmax
        
        if assim <= 0:
            net_assim = -999999999.9
        else:
            net_assim = -(cost_transp + cost_vcmax + cost_jmax) / assim
        
        if maximize:
            net_assim = -net_assim
            
        if return_all:
            return {
                'vcmax': vcmax, 'jmax': jmax, 'gs': gs, 'ci': ci, 'chi': ci/ca,
                'a_c': a_c, 'a_j': a_j, 'assim': assim, 'ci_c': ci_c, 'ci_j': ci_j,
                'cost_transp': cost_transp, 'cost_vcmax': cost_vcmax, 
                'cost_jmax': cost_jmax, 'net_assim': net_assim
            }
        else:
            return net_assim
    
    # Optimization
    result = minimize(
        optimise_this_gs_vcmax_jmax,
        x0=[vcmax_start, gs_start, jmax_start],
        args=([kmm, gammastar, ns_star, ca, vpd], (ppfd * fapar), kphio, beta, c_cost/4),
        method='L-BFGS-B',
        bounds=[(vcmax_start*0.001, vcmax_start*1000), 
                (gs_start*0.001, gs_start*1000),
                (jmax_start*0.001, jmax_start*1000)],
        options={'maxiter': 100000}
    )
    
    varlist = optimise_this_gs_vcmax_jmax(
        result.x, [kmm, gammastar, ns_star, ca, vpd], 
        (fapar * ppfd), kphio, beta, c_cost/4, 
        maximize=False, return_all=True
    )
    
    return varlist

def rpmodel(tc, vpd, co2, fapar, ppfd, patm=None, elv=None,
            kphio=None, beta=146.0, c_cost=0.41,
            soilm=1.0, meanalpha=1.0, apar_soilm=0.0, bpar_soilm=0.73300,
            c4=False, method_optci="prentice14", method_jmaxlim="wang17",
            do_ftemp_kphio=True, do_soilmstress=False, returnvar=None, verbose=False):
    """
    R implementation of the P-model and its corollary predictions
    """
    
    # Set default kphio based on conditions
    if kphio is None:
        if do_ftemp_kphio:
            kphio = 0.087182 if do_soilmstress else 0.081785
        else:
            kphio = 0.049977
    
    # Check arguments
    if patm is None and elv is None:
        raise ValueError("Provide either elevation (elv) or atmospheric pressure (patm).")
    elif patm is None:
        if verbose:
            warnings.warn("Atmospheric pressure (patm) not provided. Calculating from elevation.")
        patm = calc_patm(elv)
    
    # Fixed parameters
    c_molmass = 12.0107
    kPo = 101325.0
    kTo = 25.0
    rd_to_vcmax = 0.015
    
    # Temperature dependence of quantum yield efficiency
    ftemp_kphio_val = calc_ftemp_kphio(tc) if do_ftemp_kphio else 1.0
    
    # Soil moisture stress
    soilmstress = calc_soilmstress(soilm, meanalpha, apar_soilm, bpar_soilm) if do_soilmstress else 1.0
    
    # Photosynthesis model parameters
    ca = co2_to_ca(co2, patm)
    gammastar = calc_gammastar(tc, patm)
    kmm = calc_kmm(tc, patm)
    
    ns = calc_viscosity_h2o(tc, patm)
    ns25 = calc_viscosity_h2o(kTo, kPo)
    ns_star = ns / ns25
    
    # Optimal ci calculation
    if c4:
        out_optchi = calc_chi_c4()
    elif method_optci == "prentice14":
        out_optchi = calc_optimal_chi(kmm, gammastar, ns_star, ca, vpd, beta)
    elif method_optci == "prentice14_num":
        out_optim_num = calc_optim_num(
            kmm, gammastar, ns_star, ca, vpd, ppfd, fapar, kphio, beta, c_cost,
            vcmax_start=30.0, gs_start=0.8, jmax_start=40.0
        )
        gamma = gammastar / ca
        kappa = kmm / ca
        out_optchi = {
            'chi': out_optim_num['chi'],
            'mj': (out_optim_num['chi'] - gamma) / (out_optim_num['chi'] + 2 * gamma),
            'mc': (out_optim_num['chi'] - gamma) / (out_optim_num['chi'] + kappa),
            'mjoc': (out_optim_num['chi'] + kappa) / (out_optim_num['chi'] + 2 * gamma)
        }
        method_jmaxlim = "prentice14_num"
    else:
        raise ValueError(f"Unknown method_optci: {method_optci}")
    
    ci = out_optchi['chi'] * ca
    iwue = (ca - ci) / 1.6
    
    # Vcmax and light use efficiency
    if c4:
        out_lue_vcmax = calc_lue_vcmax_c4(kphio, ftemp_kphio_val, c_molmass, soilmstress)
    elif method_jmaxlim == "wang17":
        out_lue_vcmax = calc_lue_vcmax_wang17(out_optchi, kphio, ftemp_kphio_val, c_molmass, soilmstress, c_cost)
    elif method_jmaxlim == "smith19":
        out_lue_vcmax = calc_lue_vcmax_smith19(out_optchi, kphio, ftemp_kphio_val, c_molmass, soilmstress)
    elif method_jmaxlim == "none":
        out_lue_vcmax = calc_lue_vcmax_none(out_optchi, kphio, ftemp_kphio_val, c_molmass, soilmstress)
    elif method_jmaxlim == "prentice14_num":
        out_lue_vcmax = {
            'vcmax_unitiabs': out_optim_num['vcmax'] / (fapar * ppfd),
            'lue': c_molmass * out_optim_num['assim'] / (fapar * ppfd)
        }
    else:
        raise ValueError(f"Unknown method_jmaxlim: {method_jmaxlim}")
    
    # Temperature corrections
    ftemp25_inst_vcmax = calc_ftemp_inst_vcmax(tc, tc, tcref=25.0)
    vcmax25_unitiabs = out_lue_vcmax['vcmax_unitiabs'] / ftemp25_inst_vcmax
    
    ftemp_inst_rd_val = calc_ftemp_inst_rd(tc)
    rd_unitiabs = rd_to_vcmax * (ftemp_inst_rd_val / ftemp25_inst_vcmax) * out_lue_vcmax['vcmax_unitiabs']
    
    # Quantities that scale with absorbed light
    iabs = fapar * ppfd
    
    gpp = iabs * out_lue_vcmax['lue'] if not np.isnan(iabs) else np.nan
    vcmax = iabs * out_lue_vcmax['vcmax_unitiabs'] if not np.isnan(iabs) else np.nan
    vcmax25 = iabs * vcmax25_unitiabs if not np.isnan(iabs) else np.nan
    rd = iabs * rd_unitiabs if not np.isnan(iabs) else np.nan
    
    # Jmax calculation
    if not np.isnan(iabs):
        fact_jmaxlim = vcmax * (ci + 2.0 * gammastar) / (kphio * iabs * (ci + kmm))
        jmax = 4.0 * kphio * iabs / np.sqrt((1.0/fact_jmaxlim)**2 - 1.0)
    else:
        jmax = np.nan
    
    ftemp25_inst_jmax = calc_ftemp_inst_jmax(tc, tc, tcref=25.0)
    jmax25 = jmax / ftemp25_inst_jmax if not np.isnan(jmax) else np.nan
    
    # Assimilation rate checks
    if c4:
        a_j = kphio * iabs * out_optchi['mj'] * fact_jmaxlim if not np.isnan(iabs) else np.nan
        a_c = vcmax * out_optchi['mc'] if not np.isnan(vcmax) else np.nan
    else:
        a_j = kphio * iabs * (ci - gammastar) / (ci + 2.0 * gammastar) * fact_jmaxlim if not np.isnan(iabs) else np.nan
        a_c = vcmax * (ci - gammastar) / (ci + kmm) if not np.isnan(vcmax) else np.nan
    
    if not np.isnan(a_j) and not np.isnan(a_c) and not np.isclose(a_j, a_c, atol=0.001):
        warnings.warn(f"Light and Rubisco-limited assimilation rates differ: a_j={a_j}, a_c={a_c}")
    
    assim = min(a_j, a_c) if not np.isnan(a_j) and not np.isnan(a_c) else np.nan
    
    if not np.isnan(assim) and not np.isnan(gpp):
        assim_check = np.isclose(assim, gpp / c_molmass, atol=0.001)
        if not assim_check:
            warnings.warn("Assimilation and GPP are not identical within tolerance.")
    
    gs = assim / (ca - ci) if not np.isnan(assim) and (ca - ci) != 0 else np.nan
    
    # Output dictionary
    out = {
        'ca': ca,
        'gammastar': gammastar,
        'kmm': kmm,
        'ns_star': ns_star,
        'chi': out_optchi['chi'],
        'mj': out_optchi.get('mj'),
        'mc': out_optchi.get('mc'),
        'ci': ci,
        'lue': out_lue_vcmax['lue'],
        'gpp': gpp,
        'iwue': iwue,
        'gs': gs,
        'vcmax': vcmax,
        'vcmax25': vcmax25,
        'jmax': jmax,
        'jmax25': jmax25,
        'rd': rd
    }
    
    if returnvar is not None:
        if isinstance(returnvar, str):
            returnvar = [returnvar]
        out = {k: v for k, v in out.items() if k in returnvar}
    
    return out
