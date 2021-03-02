import os
import sys
import math
import math_ext as m
from abc import ABC, abstractmethod
import const
import ccalc_pvt
import cpipe


sys.path.append(os.path.abspath(''))

def sind(ang):
    return math.sin(ang / 180 * math.pi)

def log10(X):
    fn_return_value = math.log(X) / math.log(10)
    return fn_return_value
# ------- z-factor classes -------

class z_class_estimated(ABC):
    @abstractmethod
    def z_factor_estimated(self,tpr,ppr,z):
        pass

class z_class(ABC):
    @abstractmethod
    def z_factor(self,args_dic):
        pass

class pseudo_temp_press(ABC):
    @abstractmethod
    def press_temp(self, gamma_gas):
        pass

class pseudo_standing(pseudo_temp_press):
    def press_temp(self, gamma_gas):
        temp = 93.3 + 180 * gamma_gas - 6.94 * m.pow_(gamma_gas, 2)
        press = 4.6 + 0.1 * gamma_gas - 0.258 * m.pow_(gamma_gas, 2)
        return temp, press

class pseudo(pseudo_temp_press):
    def press_temp(self, gamma_gas):
        temp = 95 + 171 * gamma_gas
        press = 4.9 - 0.4 * gamma_gas
        return temp, press

class z_factor_estimated_ran_chuk(z_class_estimated):
    def z_factor_estimated(self,tpr,ppr,z):
        a1 = 0.3265
        a2 = -1.07
        a3 = -0.5339
        a4 = 0.01569
        a5 = -0.05165
        a6 = 0.5475
        a7 = -0.7361
        a8 = 0.1844
        a9 = 0.1056
        a10 = 0.6134
        a11 = 0.721
        rho_r = 0.27 * ppr / (z * tpr)
        rho_r_2 = m.pow_(rho_r, 2)
        tpr_2 = m.pow_(tpr, 2)
        tpr_3 = tpr_2 * tpr
        if (rho_r_2 is None or tpr == 0): return None
        result = -z + (a1 + a2 / tpr + a3 / tpr_3 + a4 / m.pow_(tpr_2, 2) + a5 / m.pow_(tpr, 5)) * rho_r + (
                    a6 + a7 / tpr + a8 / tpr_2) * rho_r_2 - a9 * (a7 / tpr + a8 / tpr_2) * m.pow_(rho_r, 5) + a10 * (
                             1 + a11 * rho_r_2) * rho_r_2 / tpr_3 * m.exp_(-a11 * rho_r_2) + 1
        return result

class z_factor_dran_chuk(z_class):
    def z_factor(self,args_dic):
        # Расчет Z-фактора, корреляция Dranchuk
        if args_dic is None or len(args_dic) == 0:
            return None
        if ('t_pseudo' in args_dic and 'p_pseudo' in args_dic) != True:
            return None
        t_pseudo = args_dic['t_pseudo']
        p_pseudo = args_dic['p_pseudo']
        if t_pseudo != t_pseudo or p_pseudo != p_pseudo or t_pseudo is None or p_pseudo is None:
            return None

        z_low = 0.1
        z_hi = 5
        i = 0

        z_cl = z_factor_estimated_ran_chuk()

        while (not (i > 20 or m.abs_(z_low - z_hi) < 0.001)):
            z_mid = 0.5 * (z_hi + z_low)
            y_low = z_cl.z_factor_estimated(t_pseudo, p_pseudo, z_low)
            y_hi = z_cl.z_factor_estimated(t_pseudo, p_pseudo, z_mid)
            if (y_low * y_hi < 0):
                z_hi = z_mid
            else:
                z_low = z_mid
            i = i + 1
        return z_mid

class z_factor_2015_kareem(z_class):
    def z_factor(self,args_dic):
        # Расчет Z-фактора, корреляция 2015_kareem
        # based on  https://link.springer.com/article/10.1007/s13202-015-0209-3
        # Kareem, L.A., Iwalewa, T.M. & Al-Marhoun, M.
        # New explicit correlation for the compressibility factor of natural gas: linearized z-factor isotherms.
        # J Petrol Explor Prod Technol 6, 481–492 (2016).
        # https://doi.org/10.1007/s13202-015-0209-3
        if args_dic is None or len(args_dic) == 0:
            return None
        if ('t_pseudo' in args_dic and 'p_pseudo' in args_dic) != True:
            return None
        t_pseudo = args_dic['t_pseudo']
        p_pseudo = args_dic['p_pseudo']
        if t_pseudo != t_pseudo or p_pseudo != p_pseudo or t_pseudo is None or p_pseudo is None:
            return None
        t = 0
        AA = 0
        BB = 0
        CC = 0
        DD = 0
        EE = 0
        FF = 0
        GG = 0
        y = 0
        z = 0
        a = [0] * 20
        a[1] = 0.317842
        a[2] = 0.382216
        a[3] = -7.768354
        a[4] = 14.290531
        a[5] = 0.000002
        a[6] = -0.004693
        a[7] = 0.096254
        a[8] = 0.16672
        a[9] = 0.96691
        a[10] = 0.063069
        a[11] = -1.966847
        a[12] = 21.0581
        a[13] = -27.0246
        a[14] = 16.23
        a[15] = 207.783
        a[16] = -488.161
        a[17] = 176.29
        a[18] = 1.88453
        a[19] = 3.05921
        DPpr = None

        t = 1 / t_pseudo
        AA = a[1] * t * math.exp(a[2] * (1 - t) ** 2) * p_pseudo
        BB = a[3] * t + a[4] * t ** 2 + a[5] * t ** 6 * p_pseudo ** 6
        CC = a[9] + a[8] * t * p_pseudo + a[7] * t ** 2 * p_pseudo ** 2 + a[6] * t ** 3 * p_pseudo ** 3
        DD = a[10] * t * math.exp(a[11] * (1 - t) ** 2)
        EE = a[12] * t + a[13] * t ** 2 + a[14] * t ** 3
        FF = a[15] * t + a[16] * t ** 2 + a[17] * t ** 3
        GG = a[18] + a[19] * t
        DPpr = DD * p_pseudo
        y = DPpr / ((1 + AA ** 2) / CC - AA ** 2 * BB / (CC ** 3))
        z = DPpr * (1 + y + y ** 2 - y ** 3) / (DPpr + EE * y ** 2 - FF * y ** GG) / ((1 - y) ** 3)
        return z

class z_factor(z_class):
    def z_factor(self,args_dic):

        a = 0
        b = 0
        c = 0
        d = 0

        if args_dic is None or len(args_dic) == 0:
            return None
        if ('t_pseudo' in args_dic and 'p_pseudo' in args_dic) != True:
            return None
        t_pseudo = args_dic['t_pseudo']
        p_pseudo = args_dic['p_pseudo']
        if t_pseudo != t_pseudo or p_pseudo != p_pseudo or t_pseudo is None or p_pseudo is None:
            return None

        a = 1.39 * m.pow_((t_pseudo - 0.92), 0.5) - 0.36 * t_pseudo - 0.101
        b = p_pseudo * (0.62 - 0.23 * p_pseudo) + p_pseudo * p_pseudo * (0.006 / (t_pseudo - 0.86) - 0.037) + \
            0.32 * p_pseudo ^ 6 / m.exp_(20.723 * (p_pseudo - 1))
        c = 0.132 - 0.32 * m.log_(t_pseudo) / m.log_(10)
        d = m.exp_(0.715 - 1.128 * t_pseudo + 0.42 * t_pseudo * t_pseudo)
        z = a + (1 - a) * m.exp_(-b) + c * m.pow_(p_pseudo, d)

        return z

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Cash-Karp adaptive ODE solver.
#
#This subroutine solves ODE  Y#=f(Y,x)  with  initial  conditions  Y(xs)=Ys
#(here Y may be single variable or vector of N variables).
#
#INPUT PARAMETERS:
#    Y       -   initial conditions, array[0..N-1].
#                contains values of Y[] at X[0]
#    N       -   system size
#    X       -   points at which Y should be tabulated, array[0..M-1]
#                integrations starts at X[0], ends at X[M-1],  intermediate
#                values at X[i] are returned too.
#                SHOULD BE ORDERED BY ASCENDING OR BY DESCENDING!!!!
#    M       -   number of intermediate points + first point + last point:
#                * M>2 means that you need both Y(X[M-1]) and M-2 values at
#                  intermediate points
#                * M=2 means that you want just to integrate from  X[0]  to
#                  X[1] and don#t interested in intermediate values.
#                * M=1 means that you don#t want to integrate :)
#                  it is degenerate case, but it will be handled correctly.
#                * M<1 means error
#    Eps     -   tolerance (absolute/relative error on each  step  will  be
#                less than Eps). When passing:
#                * Eps>0, it means desired ABSOLUTE error
#                * Eps<0, it means desired RELATIVE error.  Relative errors
#                  are calculated with respect to maximum values of  Y seen
#                  so far. Be careful to use this criterion  when  starting
#                  from Y[] that are close to zero.
#    H       -   initial  step  lenth,  it  will  be adjusted automatically
#                after the first  step.  If  H=0,  step  will  be  selected
#                automatically  (usualy  it  will  be  equal  to  0.001  of
#                min(x[i]-x[j])).
#
#OUTPUT PARAMETERS
#    State   -   structure which stores algorithm state between  subsequent
#                calls of OdeSolverIteration. Used for reverse communication.
#                This structure should be passed  to the OdeSolverIteration
#                subroutine.
#
#SEE ALSO
#    AutoGKSmoothW, AutoGKSingular, AutoGKIteration, AutoGKResults.
#
#
#  -- ALGLIB --
#     Copyright 01.09.2009 by Bochkanov Sergey
#
##########################################################################
#
ODESolverMaxGrow = 3
ODESolverMaxShrink = 10
class RState:
    Stage = -1
    IA = [0]*6
    BA = [None]
    RA = [0]*6
    CA = None

class State:
    rState = RState()
    DY = None
    Eps = None
    EScale = None
    FracEps = None
    H = None
    M = None
    N = None
    RepNFEV = None
    RepTerminationType = None
    RKA = None
    RKB = None
    RKC = None
    RKCS = None
    RKK = None
    solverType = None
    X = None
    XG = []
    XScale = 1
    y = None
    YC = []
    YN = None
    YNS = None
    YTbl = None

def ODESolverInit(_solverType, y_arr, N, X_arr, M, eps, h):
    # Internal initialization subroutine
    state = State()
    #
    # check parameters.
    #
    if N <= 0 or M < 1 or eps == 0:
        state.RepTerminationType = -1
        return state
    if h < 0:
        h = -h
    #
    # quick exit if necessary.
    # after this block we assume that M>1
    #
    if M == 1:
        state.rState.RepNFEV = 0
        state.RepTerminationType = 1
        state.YTbl[0] = []
        for yi in y_arr:
            state.YTbl[0].append(yi)

        state.XG = []
        for xi in X_arr:
            state.XG.append(xi)
        return state

    #
    # check again: correct order of X[]
    #
    #If X(1#) = X(0#) Then
    #State.RepTerminationType = -2#
    #Exit Sub
    #End If
    #For i = 1# To M - 1# Step 1
    #If X(1#) > X(0#) And X(i) <= X(i - 1#) Or X(1#) < X(0#) And X(i) >= X(i - 1#) Then
    #State.RepTerminationType = -2#
    #Exit Sub
    #End If
    #Next i

    #
    # auto-select H if necessary
    #
    if h is None or h == 0:
        V = math.abs(X_arr[1] - X_arr[0])
        if M > 2:
            for i in range(2, M):
                V = math.min(V, math.abs(X_arr[i] - X_arr(i - 1)))
            h = 0.001 * V
    #
    # store parameters
    #
    state.N = N
    state.M = M
    state.H = h
    state.Eps = abs(eps)
    state.FracEps = eps < 0
    state.XG = []
    for xi in X_arr:
        state.XG.append(xi)

    if X_arr[1] > X_arr[0]:
        state.XScale = 1
    else:
        state.XScale = -1
        for xi in X_arr:
            xi = - xi
    state.YC = []
    for yi in y_arr:
        state.YC.append(yi)

    state.RepTerminationType = 0
    state.solverType = _solverType
    return state

def h_l_arr_theta_deg(flow_pattern, lambda_l, n_fr, n_lv, arr_theta_deg, Payne_et_all):
    #function calculating liquid holdup
    #flow_pattern - flow pattern (0 -Segregated, 1 - Intermittent, 2 - Distributed)
    #lambda_l - volume fraction of liquid at no-slip conditions
    #n_fr - Froude number
    #n_lv - liquid velocity number
    #arr_theta_deg - pipe inclination angle, (Degrees)
    #payne_et_all - flag indicationg weather to applied Payne et all correction for holdup (0 - not applied, 1 - applied)
    #Constants to determine liquid holdup
    a = [0.98, 0.845, 1.065]
    b = [0.4846, 0.5351, 0.5824]
    c = [0.0868, 0.0173, 0.0609]
    #constants to determine liquid holdup correction
    E = [0.011, 2.96, 1]
    F = [-3.768, 0.305, 0]
    g = [3.539, -0.4473, 0]
    h = [ -1.614, 0.0978, 0]
    #calculate liquid holdup at no slip conditions

    h_l_0 = a[flow_pattern] * lambda_l ** b[flow_pattern] / (n_fr ** c[flow_pattern])

    #calculate correction for inclination angle
    CC = max(0, (1 - lambda_l) * math.log(E[flow_pattern] * lambda_l ** F[flow_pattern] * n_lv ** g[flow_pattern] * n_fr ** h[flow_pattern]))

    #convert angle to radians

    arr_theta_deg_d = math.pi / 180 * arr_theta_deg
    psi = 1 + CC * (math.sin(1.8 * arr_theta_deg_d) + 0.333 * (math.sin(1.8 * arr_theta_deg_d)) ** 3)

    #calculate liquid holdup with payne et al. correction factor
    h_l_arr_theta_deg = None
    if Payne_et_all > 0:
        if arr_theta_deg > 0: #uphill flow
            h_l_arr_theta_deg = max(min(1, 0.924 * h_l_0 * psi), lambda_l)
        else:  #downhill flow
            h_l_arr_theta_deg = max(min(1, 0.685 * h_l_0 * psi), lambda_l)
    else:
        h_l_arr_theta_deg = max(min(1, h_l_0 * psi), lambda_l)
    return h_l_arr_theta_deg

def calc_friction_factor(n_re, E, Rough_pipe):
    f_n = None
    f_n_new = None
    f_int = None
    i = None
    #Calculates friction factor given pipe relative roughness and Reinolds number
    #Parameters
    #n_re - Reinolds number
    #e - pipe relative roughness
    #Rough_pipe - flag indicating weather to calculate friction factor for rough pipe using Moody correlation (Rough_pipe > 0), or
    #using Drew correlation for smooth pipes
    #friction factor and iterated friction factor
    if n_re == 0:
        f_n = 0
    elif n_re > 2000: # turbulent flow
        if Rough_pipe > 0: # 'calculate friction factor for rough pipes according to Moody method - Payne et all modification for Beggs&Brill correlation
            f_n = (2 * log10(2 / 3.7 * E - 5.02 / n_re * log10(2 / 3.7 * E + 13 / n_re)) ) ** - 2
            i = 0
            while 1:
                f_n_new = (1.74 - 2 * log10(2 * E + 18.7 /  (n_re * f_n ** 0.5 )) ) ** - 2
                i = i + 1
                f_int = f_n
                f_n = f_n_new
                #stop when error is sufficiently small or max number of iterations exceedied
                if (abs(f_n_new - f_int) <= 0.001 or i > 19):
                    break
        else:
            f_n = 0.0056 + 0.5 * n_re ** - 0.32
    else:
        f_n = 64 / n_re
    fn_return_value = f_n
    return fn_return_value

def BegsBrillGradient(arr_d_m, arr_theta_deg, eps_m, Ql_rc_m3day, Qg_rc_m3day, Mul_rc_cP, Mug_rc_cP, sigma_l_Nm, rho_lrc_kgm3, rho_grc_kgm3, Payne_et_all_holdup=0, Payne_et_all_friction=1, c_calibr_grav=1, c_calibr_fric=1):
    roughness_d = None
    dPdLg_out_atmm = None
    dPdLf_out_atmm = None
    Hl_out_fr = None
    fpat_out_num = None
    dPdLa_out_atmm = None
    Ap_m2 = None
    lambda_l = None
    Vsl_msec = None
    Vsg_msec = None
    Vsm_msec = None
    Rho_n_kgm3 = None
    rho_s = None
    Mu_n_cP = None
    n_re = None
    n_fr = None
    n_lv = None
    flow_pattern = None
    l_2 = None
    l_3 = None
    AA = None
    f_n = None
    F = None
    y = None
    S = None
    c_p = 0.000009871668
    #function for calculation of pressure gradient in pipe according to Begs and Brill method
    #Return (psi/ft (atma/m))
    #Arguments
    #d - pipe internal diameter ( (m))
    #arr_theta_deg - pipe inclination angel (degrees)
    #eps_m - pipe wall roughness ( (m))
    #p - reference pressure ( (atma))
    #q_oSC - oil rate at standard conditions ( (m3/day))
    #q_wSC - water rate at standard conditions ( (m3/day))
    #q_gSC - gas rate at standard conditions ((m3/day))
    #Bo_m3m3 - oil formation volume factor at reference pressure ( (m3/sm3))
    #Bw_m3m3 - water formation volume factor at reference pressure ( (m3/sm3))
    #Bg_m3m3 - gas formation volume factorat reference pressure ( (m3/sm3))
    #rs - gas-oil solution ratio at reference pressure ( (sm3/sm3))
    #mu_oil_cP - oil viscosity at reference pressure (cp)
    #mu_wat_cP - water viscosity at reference pressure (cp)
    #mu_gas_cP - gAs viscosity at reference pressure (cp)
    #sigma_oil_gas_Nm - oil-gAs surface tension coefficient ((Newton/m))
    #sigma_wat_gas_Nm - water-gAs surface tension coefficient ( (Newton/m))
    #rho_oSC - oil density at standard conditions ( (kg/m3))
    #rho_wSC - water density at standard conditions ( (kg/m3))
    #rho_gSC - gas density at standard conditions((kg/m3))
    #
    #Payne_et_all_holdup - flag indicationg weather to applied Payne et all correction and holdup (0 - not applied, 1 - applied)
    #Payne_et_all_friction - flag indicationg weather to apply Payne et all correction for friction (0 - not applied, 1 - applied)
    #dpdl_g - used to otput pressure gradient due to gravity ( (atma/m))
    #dpdl_f - used to output pressure gradient due to friction ( (atma/m))
    #v_sl - used to output liquid superficial velocity ( (m/sec))
    #v_sg - used to output gas superficial velocity ( (m/sec))
    #h_l - used to output liquid holdup
    #Calculate auxilary values

    const_conver_day_sec = 86400
    const_conver_sec_day = 1 / const_conver_day_sec
    const_g = 9.81

    Ap_m2 = math.pi * arr_d_m ** 2 / 4
    if Ql_rc_m3day == 0:
        # специально отработать случай нулевого дебита
        lambda_l = 1
        Hl_out_fr = 1
        F = 0
        Rho_n_kgm3 = rho_lrc_kgm3 * lambda_l + rho_grc_kgm3 * (1 - lambda_l )
        flow_pattern = 0
    else:
        lambda_l = Ql_rc_m3day /  ( Ql_rc_m3day + Qg_rc_m3day )
        roughness_d = eps_m / arr_d_m
        Vsl_msec = const_conver_sec_day * Ql_rc_m3day / Ap_m2
        Vsg_msec = const_conver_sec_day * Qg_rc_m3day / Ap_m2
        Vsm_msec = Vsl_msec + Vsg_msec
        Rho_n_kgm3 = rho_lrc_kgm3 * lambda_l + rho_grc_kgm3 * (1 - lambda_l )
        Mu_n_cP = Mul_rc_cP * lambda_l + Mug_rc_cP * ( 1 - lambda_l )
        n_re = 1000 * Rho_n_kgm3 * Vsm_msec * arr_d_m / Mu_n_cP
        n_fr = Vsm_msec ** 2 / (const_g * arr_d_m)
        n_lv = Vsl_msec * ( rho_lrc_kgm3 / (const_g * sigma_l_Nm)) ** 0.25
        #-----------------------------------------------------------------------
        #determine flow pattern
        if ( n_fr >= 316 * lambda_l ** 0.302 or n_fr >= 0.5 * lambda_l ** - 6.738 ):
            flow_pattern = 2
        else:
            if ( n_fr <= 0.000925 * lambda_l ** - 2.468):
                flow_pattern = 0
            else:
                if ( n_fr <= 0.1 * lambda_l ** - 1.452):
                    flow_pattern = 3
                else:
                    flow_pattern = 1
        #-----------------------------------------------------------------------
        #determine liquid holdup
        if ( flow_pattern == 0 or flow_pattern == 1 or flow_pattern == 2):
            Hl_out_fr = h_l_arr_theta_deg(flow_pattern, lambda_l, n_fr, n_lv, arr_theta_deg, Payne_et_all_holdup)
        else:
            l_2 = 0.000925 * lambda_l ** - 2.468
            l_3 = 0.1 * lambda_l ** - 1.452
            AA = (l_3 - n_fr) / (l_3 - l_2)
            Hl_out_fr = AA * h_l_arr_theta_deg(0, lambda_l, n_fr, n_lv, arr_theta_deg, Payne_et_all_holdup) + ( 1 - AA ) * h_l_arr_theta_deg(1, lambda_l, n_fr, n_lv, arr_theta_deg, Payne_et_all_holdup)
        #Calculate normalized friction factor
        f_n = calc_friction_factor(n_re, roughness_d, Payne_et_all_friction)
        #calculate friction factor correction for multiphase flow
        y = max(lambda_l / Hl_out_fr ** 2, 0.001)
        if (y > 1 and y < 1.2):
            S = math.log(2.2 * y - 1.2)
        else:
            S = math.log(y) / (- 0.0523 + 3.182 * math.log(y) - 0.8725 * (math.log(y)) ** 2 + 0.01853 * (math.log(y)) ** 4)
        #calculate friction factor
        F = f_n * math.exp(S)
    #calculate mixture density
    rho_s = rho_lrc_kgm3 * Hl_out_fr + rho_grc_kgm3 * (1 - Hl_out_fr)
    #calculate pressure gradient due to gravity
    dPdLg_out_atmm = c_p * rho_s * const_g * sind(arr_theta_deg)
    #calculate pressure gradient due to friction
    dPdLf_out_atmm = c_p * F * Rho_n_kgm3 * Vsm_msec ** 2 / (2 * arr_d_m)
    #calculate pressure gradient
    dPdLa_out_atmm = 0
    # Hl_out_fr рассчитано по ходу дела
    fpat_out_num = flow_pattern
    fn_return_value = [dPdLg_out_atmm * c_calibr_grav + dPdLf_out_atmm * c_calibr_fric, dPdLg_out_atmm * c_calibr_grav, dPdLf_out_atmm * c_calibr_fric, dPdLa_out_atmm, Vsl_msec, Vsg_msec, Hl_out_fr, fpat_out_num]
    return fn_return_value

#=================================================================================================
# новый подход - можно обойтись без разделение на прямой участок и кривой - расчет за один проход.
#================================================================================================
#
def calc_grad(l_m, p_atma, t_C, paramCalc, d_m, theta_deg, angle_hmes_deg, rough_m, Qliq, Fw, aPVTdic, calc_dtdl=True, pcas_atma=0.95):
    # l_m - здесь была нужна только для определения диаметра по глубине
    dp_dl= None

    #проверим на корректность исходных данных
    if p_atma < const.minPpipe_atma:
        dp_dl = 0
        return 0
    c_calibr_grav = 1
    c_calibr_fric = 1

    dpdlg_out= None
    dpdlf_out= None
    dpdla_out= None
    v_sl_out= None
    v_sg_out= None
    vl_msec= None
    vg_msec= None
    h_l_out= None
    fpat_out = None
    theta_deg= None
    theta_sign = None
    dp_dl_arr = None
    dt_dl= None
    v= None
    dvdL= None
    # функция расчета градиента давления и температуры в скважине при заданных параметрах
    # возвращает все параметры потока в заданной точке трубы при заданых термобарических условиях.
    #  L_m      - измеренная глубина на которой ведется расчет, нужна для привязки по температуре
    #  p_atma   - давление в заданной точке
    #  T_C      - температура в заданной точке
    #  calc_dtdl
    #  pcas_atma - затрубное давление для оптимизации расчета барботажа в затрубе

    #Allocate variables used to output auxilary values
    if paramCalc.FlowAlongCoord:
        theta_sign = - 1
    else:
        theta_sign = 1
    theta_deg = theta_sign * angle_hmes_deg
    z_class = z_factor_2015_kareem()
    pseudo_class = pseudo_standing()
    PVTdic = ccalc_pvt.calc_pvt_vr(p_atma, t_C, Qliq, Fw / 100, aPVTdic['PVTcorr'], aPVTdic, z_class, pseudo_class)

    dp_dl_arr = [0]*8
    dp_dl_arr[7] = 101
#    if l_m < paramCalc.length_gas_m:
#        corr = gas
#    else:
#        corr = paramCalc.correlation
    # BeggsBrill по умолчанию пока, корреляции в будущем передавать функциями
    if (paramCalc.correlation == 0):
        dp_dl_arr = BegsBrillGradient(d_m, theta_deg, rough_m, PVTdic['qliq_rc_m3day'], PVTdic['q_gas_rc_m3day'], PVTdic['mu_liq_cP'],
                                    PVTdic['mu_gas_cP'], PVTdic['sigma_liq_Nm'], PVTdic['rho_liq_rc_kgm3'], PVTdic['rho_gas_rc_kgm3'],
                                      0, 1, c_calibr_grav, c_calibr_fric)
#    elif (select_variable_0 == Ansari):
#        if p_atma > pcas_atma:
#            dp_dl_arr = unf_AnsariGradient(d_m, theta_deg, rough_m, with_variable0.qliq_rc_m3day, with_variable0.q_gas_rc_m3day, with_variable0.mu_liq_cP, with_variable0.mu_gas_cP, with_variable0.sigma_liq_Nm, with_variable0.rho_liq_rc_kgm3, with_variable0.rho_gas_rc_kgm3, p_atma, VBGetMissingArgument(unf_AnsariGradient, 11), c_calibr_grav, c_calibr_fric)
#    elif (select_variable_0 == gas):
#        if p_atma > pcas_atma:
#            dp_dl_arr = unf_GasGradient(d_m, theta_deg, rough_m, with_variable0.q_gas_rc_m3day, with_variable0.mu_gas_cP, with_variable0.rho_gas_rc_kgm3, p_atma)
#            # gas gradient do not use calibration coeficients
#    elif (select_variable_0 == Unified):
#        dp_dl_arr = unf_UnifiedTUFFPGradient(d_m, theta_deg, rough_m, with_variable0.qliq_rc_m3day, with_variable0.q_gas_rc_m3day, with_variable0.mu_liq_cP, with_variable0.mu_gas_cP, with_variable0.sigma_liq_Nm, with_variable0.rho_liq_rc_kgm3, with_variable0.rho_gas_rc_kgm3, p_atma, VBGetMissingArgument(unf_UnifiedTUFFPGradient, 11), c_calibr_grav, c_calibr_fric)
#    elif (select_variable_0 == Gray):
#        dp_dl_arr = unf_GrayModifiedGradient(d_m, theta_deg, rough_m, with_variable0.qliq_rc_m3day, with_variable0.q_gas_rc_m3day, with_variable0.mu_liq_cP, with_variable0.mu_gas_cP, with_variable0.sigma_liq_Nm, with_variable0.rho_liq_rc_kgm3, with_variable0.rho_gas_rc_kgm3, 0, 1, VBGetMissingArgument(unf_GrayModifiedGradient, 12), c_calibr_grav, c_calibr_fric)
#    elif (select_variable_0 == HagedornBrown):
#        dp_dl_arr = unf_HagedornandBrawnmodified(d_m, theta_deg, rough_m, with_variable0.qliq_rc_m3day, with_variable0.q_gas_rc_m3day, with_variable0.mu_liq_cP, with_variable0.mu_gas_cP, with_variable0.sigma_liq_Nm, with_variable0.rho_liq_rc_kgm3, with_variable0.rho_gas_rc_kgm3, p_atma, 0, 1, VBGetMissingArgument(unf_HagedornandBrawnmodified, 13), c_calibr_grav, c_calibr_fric)
#    elif (select_variable_0 == SakharovMokhov):
#       dp_dl_arr = unf_Saharov_Mokhov_Gradient(d_m, theta_deg, rough_m, p_atma, with_variable0.q_oil_sm3day, with_variable0.q_wat_sm3day, with_variable0.q_gas_sm3day, with_variable0.bo_m3m3, with_variable0.bw_m3m3, with_variable0.bg_m3m3, with_variable0.rs_m3m3, with_variable0.mu_oil_cP, with_variable0.mu_wat_cP, with_variable0.mu_gas_cP, with_variable0.sigma_oil_gas_Nm, with_variable0.sigma_wat_gas_Nm, with_variable0.rho_oil_sckgm3, with_variable0.rho_wat_sckgm3, with_variable0.rho_gas_sckgm3, VBGetMissingArgument(unf_Saharov_Mokhov_Gradient, 19), VBGetMissingArgument(unf_Saharov_Mokhov_Gradient, 20), VBGetMissingArgument(unf_Saharov_Mokhov_Gradient, 21), c_calibr_grav, c_calibr_fric)
    dp_dl = theta_sign * dp_dl_arr[0]
# Далее пока не нужно
#     dpdlg_out = theta_sign * dp_dl_arr[1]
#     dpdlf_out = theta_sign * dp_dl_arr[2]
#     dpdla_out = theta_sign * dp_dl_arr[3]
#     v_sl_out = dp_dl_arr[4]
#     v_sg_out = dp_dl_arr[5]
#     h_l_out = dp_dl_arr[6]
#     fpat_out = dp_dl_arr[7]
#     vl_msec = v_sl_out * math.pi * d_m ** 2 / 4
#     vg_msec = v_sg_out * math.pi * d_m ** 2 / 4
    # По умолчанию StartEndTemp dt_dl = dTdL_linear_Cm(Hv_m) и всегда ноль, т.к. нет разницы температур
    # для оценки температуры оценим скорость потока и ускорение
    # теперь зададим изменение температуры в потоке
    # if calc_dtdl:
    #     select_variable_1 = param.temp_method
    #     if (select_variable_1 == StartEndTemp):
    #         dt_dl = dTdL_linear_Cm(Hv_m)
    #     elif (select_variable_1 == GeoGradTemp):
    #         dt_dl = dTdL_amb_Cm(Hv_m)
    #     elif (select_variable_1 == AmbientTemp):
    #         v = vg_msec
    #         dvdL = - v / p_atma * dp_dl
    #         dt_dl = ambient_formation.calc_dtdl_Cm(Hv_m, sind(theta_deg), t_C, with_variable0.wm_kgsec, with_variable0.cmix_JkgC, dp_dl, v, dvdL, with_variable0.cJT_Katm, not paramCalc.FlowAlongCoord)
    dt_dl = 0
    # тут надо записать в результаты все расчетные параметры
    # res.md_m = l_m
    # res.vd_m = Hv_m
    # res.dpdl_a_atmm = dpdla_out
    # res.dpdl_f_atmm = dpdlf_out
    # res.dpdl_g_atmm = dpdlg_out
    # res.fpat = fpat_out
    # res.gasfrac = fluid.gas_fraction_d()
    # res.h_l_d = h_l_out
    # res.Qg_m3day = fluid.q_gas_rc_m3day
    # res.p_atma = p_atma
    # res.t_C = t_C
    # res.v_sl_msec = v_sl_out
    # res.v_sg_msec = v_sg_out
    # res.thete_deg = theta_deg
    # res.roughness_m = rough_m
    # res.rs_m3m3 = fluid.rs_m3m3
    # res.gasfrac = fluid.gas_fraction_d
    # res.mu_oil_cP = fluid.mu_oil_cP
    # res.mu_wat_cP = fluid.mu_wat_cP
    # res.mu_gas_cP = fluid.mu_gas_cP
    # res.mu_mix_cP = fluid.mu_mix_cP
    # res.Rhoo_kgm3 = fluid.rho_oil_rc_kgm3
    # res.Rhow_kgm3 = fluid.rho_wat_rc_kgm3
    # res.rhol_kgm3 = fluid.rho_liq_rc_kgm3
    # res.Rhog_kgm3 = fluid.rho_gas_rc_kgm3
    # res.rhomix_kgm3 = fluid.rho_mix_rc_kgm3
    # res.q_oil_m3day = fluid.q_oil_rc_m3day
    # res.qw_m3day = fluid.q_wat_rc_m3day
    # res.Qg_m3day = fluid.q_gas_rc_m3day
    # res.mo_kgsec = fluid.mo_kgsec
    # res.mw_kgsec = fluid.mw_kgsec
    # res.mg_kgsec = fluid.mg_kgsec
    # res.vl_msec = vl_msec
    # res.vg_msec = vg_msec
    # res.dp_dl = dp_dl
    # res.dt_dl = dt_dl

    return dp_dl

def calc_grad_pvtcls(l_m, p_atma, t_C, paramCalc, d_m, theta_deg, angle_hmes_deg, rough_m, Qliq, Fw, class_pvt, calc_dtdl=True, pcas_atma=0.95):
    # l_m - здесь была нужна только для определения диаметра по глубине
    dp_dl= None

    #проверим на корректность исходных данных
    if p_atma < const.minPpipe_atma:
        dp_dl = 0
        return 0
    c_calibr_grav = 1
    c_calibr_fric = 1

    dpdlg_out= None
    dpdlf_out= None
    dpdla_out= None
    v_sl_out= None
    v_sg_out= None
    vl_msec= None
    vg_msec= None
    h_l_out= None
    fpat_out = None
    theta_deg= None
    theta_sign = None
    dp_dl_arr = None
    dt_dl= None
    v= None
    dvdL= None
    # функция расчета градиента давления и температуры в скважине при заданных параметрах
    # возвращает все параметры потока в заданной точке трубы при заданых термобарических условиях.
    #  L_m      - измеренная глубина на которой ведется расчет, нужна для привязки по температуре
    #  p_atma   - давление в заданной точке
    #  T_C      - температура в заданной точке
    #  calc_dtdl
    #  pcas_atma - затрубное давление для оптимизации расчета барботажа в затрубе

    #Allocate variables used to output auxilary values
    if paramCalc.FlowAlongCoord:
        theta_sign = - 1
    else:
        theta_sign = 1
    theta_deg = theta_sign * angle_hmes_deg
    z_class = z_factor_2015_kareem()
    pseudo_class = pseudo_standing()
#    PVTdic = ccalc_pvt.calc_pvt_vr(p_atma, t_C, Qliq, Fw / 100, class_pvt.PVTcorr, class_pvt, z_class, pseudo_class)

    class_pvt.calc(p_atma, t_C, Qliq, Fw / 100, z_class, pseudo_class)

    dp_dl_arr = [0]*8
    dp_dl_arr[7] = 101
#    if l_m < paramCalc.length_gas_m:
#        corr = gas
#    else:
#        corr = paramCalc.correlation
    # BeggsBrill по умолчанию пока, корреляции в будущем передавать функциями
    if (paramCalc.correlation == 0):
        dp_dl_arr = BegsBrillGradient(d_m, theta_deg, rough_m, class_pvt.qliq_rc_m3day, class_pvt.q_gas_rc_m3day, class_pvt.mu_liq_cP,
                                    class_pvt.mu_gas_cP, class_pvt.sigma_liq_Nm, class_pvt.rho_liq_rc_kgm3, class_pvt.rho_gas_rc_kgm3,
                                      0, 1, c_calibr_grav, c_calibr_fric)
#    elif (select_variable_0 == Ansari):
#        if p_atma > pcas_atma:
#            dp_dl_arr = unf_AnsariGradient(d_m, theta_deg, rough_m, with_variable0.qliq_rc_m3day, with_variable0.q_gas_rc_m3day, with_variable0.mu_liq_cP, with_variable0.mu_gas_cP, with_variable0.sigma_liq_Nm, with_variable0.rho_liq_rc_kgm3, with_variable0.rho_gas_rc_kgm3, p_atma, VBGetMissingArgument(unf_AnsariGradient, 11), c_calibr_grav, c_calibr_fric)
#    elif (select_variable_0 == gas):
#        if p_atma > pcas_atma:
#            dp_dl_arr = unf_GasGradient(d_m, theta_deg, rough_m, with_variable0.q_gas_rc_m3day, with_variable0.mu_gas_cP, with_variable0.rho_gas_rc_kgm3, p_atma)
#            # gas gradient do not use calibration coeficients
#    elif (select_variable_0 == Unified):
#        dp_dl_arr = unf_UnifiedTUFFPGradient(d_m, theta_deg, rough_m, with_variable0.qliq_rc_m3day, with_variable0.q_gas_rc_m3day, with_variable0.mu_liq_cP, with_variable0.mu_gas_cP, with_variable0.sigma_liq_Nm, with_variable0.rho_liq_rc_kgm3, with_variable0.rho_gas_rc_kgm3, p_atma, VBGetMissingArgument(unf_UnifiedTUFFPGradient, 11), c_calibr_grav, c_calibr_fric)
#    elif (select_variable_0 == Gray):
#        dp_dl_arr = unf_GrayModifiedGradient(d_m, theta_deg, rough_m, with_variable0.qliq_rc_m3day, with_variable0.q_gas_rc_m3day, with_variable0.mu_liq_cP, with_variable0.mu_gas_cP, with_variable0.sigma_liq_Nm, with_variable0.rho_liq_rc_kgm3, with_variable0.rho_gas_rc_kgm3, 0, 1, VBGetMissingArgument(unf_GrayModifiedGradient, 12), c_calibr_grav, c_calibr_fric)
#    elif (select_variable_0 == HagedornBrown):
#        dp_dl_arr = unf_HagedornandBrawnmodified(d_m, theta_deg, rough_m, with_variable0.qliq_rc_m3day, with_variable0.q_gas_rc_m3day, with_variable0.mu_liq_cP, with_variable0.mu_gas_cP, with_variable0.sigma_liq_Nm, with_variable0.rho_liq_rc_kgm3, with_variable0.rho_gas_rc_kgm3, p_atma, 0, 1, VBGetMissingArgument(unf_HagedornandBrawnmodified, 13), c_calibr_grav, c_calibr_fric)
#    elif (select_variable_0 == SakharovMokhov):
#       dp_dl_arr = unf_Saharov_Mokhov_Gradient(d_m, theta_deg, rough_m, p_atma, with_variable0.q_oil_sm3day, with_variable0.q_wat_sm3day, with_variable0.q_gas_sm3day, with_variable0.bo_m3m3, with_variable0.bw_m3m3, with_variable0.bg_m3m3, with_variable0.rs_m3m3, with_variable0.mu_oil_cP, with_variable0.mu_wat_cP, with_variable0.mu_gas_cP, with_variable0.sigma_oil_gas_Nm, with_variable0.sigma_wat_gas_Nm, with_variable0.rho_oil_sckgm3, with_variable0.rho_wat_sckgm3, with_variable0.rho_gas_sckgm3, VBGetMissingArgument(unf_Saharov_Mokhov_Gradient, 19), VBGetMissingArgument(unf_Saharov_Mokhov_Gradient, 20), VBGetMissingArgument(unf_Saharov_Mokhov_Gradient, 21), c_calibr_grav, c_calibr_fric)
    dp_dl = theta_sign * dp_dl_arr[0]
# Далее пока не нужно
#     dpdlg_out = theta_sign * dp_dl_arr[1]
#     dpdlf_out = theta_sign * dp_dl_arr[2]
#     dpdla_out = theta_sign * dp_dl_arr[3]
#     v_sl_out = dp_dl_arr[4]
#     v_sg_out = dp_dl_arr[5]
#     h_l_out = dp_dl_arr[6]
#     fpat_out = dp_dl_arr[7]
#     vl_msec = v_sl_out * math.pi * d_m ** 2 / 4
#     vg_msec = v_sg_out * math.pi * d_m ** 2 / 4
    # По умолчанию StartEndTemp dt_dl = dTdL_linear_Cm(Hv_m) и всегда ноль, т.к. нет разницы температур
    # для оценки температуры оценим скорость потока и ускорение
    # теперь зададим изменение температуры в потоке
    # if calc_dtdl:
    #     select_variable_1 = param.temp_method
    #     if (select_variable_1 == StartEndTemp):
    #         dt_dl = dTdL_linear_Cm(Hv_m)
    #     elif (select_variable_1 == GeoGradTemp):
    #         dt_dl = dTdL_amb_Cm(Hv_m)
    #     elif (select_variable_1 == AmbientTemp):
    #         v = vg_msec
    #         dvdL = - v / p_atma * dp_dl
    #         dt_dl = ambient_formation.calc_dtdl_Cm(Hv_m, sind(theta_deg), t_C, with_variable0.wm_kgsec, with_variable0.cmix_JkgC, dp_dl, v, dvdL, with_variable0.cJT_Katm, not paramCalc.FlowAlongCoord)
    dt_dl = 0
    # тут надо записать в результаты все расчетные параметры
    # res.md_m = l_m
    # res.vd_m = Hv_m
    # res.dpdl_a_atmm = dpdla_out
    # res.dpdl_f_atmm = dpdlf_out
    # res.dpdl_g_atmm = dpdlg_out
    # res.fpat = fpat_out
    # res.gasfrac = fluid.gas_fraction_d()
    # res.h_l_d = h_l_out
    # res.Qg_m3day = fluid.q_gas_rc_m3day
    # res.p_atma = p_atma
    # res.t_C = t_C
    # res.v_sl_msec = v_sl_out
    # res.v_sg_msec = v_sg_out
    # res.thete_deg = theta_deg
    # res.roughness_m = rough_m
    # res.rs_m3m3 = fluid.rs_m3m3
    # res.gasfrac = fluid.gas_fraction_d
    # res.mu_oil_cP = fluid.mu_oil_cP
    # res.mu_wat_cP = fluid.mu_wat_cP
    # res.mu_gas_cP = fluid.mu_gas_cP
    # res.mu_mix_cP = fluid.mu_mix_cP
    # res.Rhoo_kgm3 = fluid.rho_oil_rc_kgm3
    # res.Rhow_kgm3 = fluid.rho_wat_rc_kgm3
    # res.rhol_kgm3 = fluid.rho_liq_rc_kgm3
    # res.Rhog_kgm3 = fluid.rho_gas_rc_kgm3
    # res.rhomix_kgm3 = fluid.rho_mix_rc_kgm3
    # res.q_oil_m3day = fluid.q_oil_rc_m3day
    # res.qw_m3day = fluid.q_wat_rc_m3day
    # res.Qg_m3day = fluid.q_gas_rc_m3day
    # res.mo_kgsec = fluid.mo_kgsec
    # res.mw_kgsec = fluid.mw_kgsec
    # res.mg_kgsec = fluid.mg_kgsec
    # res.vl_msec = vl_msec
    # res.vg_msec = vg_msec
    # res.dp_dl = dp_dl
    # res.dt_dl = dt_dl

    return dp_dl

x_gr = 0
y_gr = 1

def calc_rs_bo_m3m3(Pintake_, Tintake_, Qliq_, Fw_, class_pvt, z_class, pseudo_class):

    class_pvt.calc(Pintake_, Tintake_, Qliq_, Fw_, z_class, pseudo_class)
    clc_rs_m3m3 = class_pvt.rs_m3m3
    clc_bo_m3m3 = class_pvt.bo_m3m3

    return clc_rs_m3m3, clc_bo_m3m3

def getFirstPointNo(Rpnew, k, array):
    i = 0
    F = True

    while F:
        F = False
        if i < k - 1:
            if Rpnew > array[i]:
                i = i + 1
                F = True

    if i == 0:
        i = 1

    gFPNo = i - 1
    return gFPNo

def getPoint(Rpnew, k, array_func1, array_func2):

    """
    getPoint возращается два значения в точках Rpnew (два графика - две точки Rpnew с разными значениями) при помощи линейной
    интерполяции.
    """
    n = 0
    X1 = 0
    X2 = 0
    y1 = 0
    y2 = 0
# для первого "графика"
    """
    getFirstPointNo предназначена для нахождения n -- индекса в массиве по точкам "X" - четные элементы массива
    чтобы расположить Rpnew между точками  Х1 и Х2 так, чтобы выполнялось условие Х1 < Rpnew < Х2
    """
    n = getFirstPointNo(Rpnew, k, array_func1[x_gr])
    X1 = array_func1[x_gr][n]
    y1 = array_func1[y_gr][n]

    if k > 1:
        X2 = array_func1[x_gr][n + 1]
        y2 = array_func1[y_gr][n + 1]
    else:
        X2 = X1
        y2 = y1

    # делаем проверку - если функция ступенчатая то выдаем не интерполированное значение, а значение в предущей точке
    if Rpnew >= X2:
        pb_a = y2
    else:
        pb_a = y1

    pb_a = (y2 - y1) / (X2 - X1) * (Rpnew - X1) + y1

# для второго "графика"
    n = 0
    X1 = 0
    X2 = 0
    y1 = 0
    y2 = 0
    n = getFirstPointNo(Rpnew, k, array_func2[x_gr])  # данная функция предназначена для поиска двух точек на между которами Rpnew
    X1 = array_func2[x_gr][n]
    y1 = array_func2[y_gr][n]

    if k > 1:
        X2 = array_func2[x_gr][n + 1]
        y2 = array_func2[y_gr][n + 1]
    else:
        X2 = X1
        y2 = y1

    # делаем проверку - если функция ступенчатая то выдаем не интерполированное значение, а значение в предущей точке
    if Rpnew >= X2:
        bob = y2
    else:
        bob = y1

    bob = (y2 - y1) / (X2 - X1) * (Rpnew - X1) + y1
    return pb_a, bob

def q_mix_rc_m3day(Pintake_, Tintake_, Qliq_, Fw_, class_pvt, z_class, pseudo_class):

    n = 10
#  промежуточные переменные для сохранения значений этих параметров
#  эти параметры должны возвращаться с теми значениями, с которыми они пришли
    rp_m3m3_r = class_pvt.rp_m3m3
    rsb_m3m3_r = class_pvt.rsb_m3m3
    bob_m3m3_r = class_pvt.bob_m3m3
    pb_atma_r = class_pvt.pb_atma
    rp_m3m3 = class_pvt.rp_m3m3

    class_pvt.rsb_m3m3 = class_pvt.rp_m3m3
    pb_atma = class_pvt.pb_atma
    Ksep = class_pvt.ksep_fr
    Rs = 0
    Bo = 0
    Rpnew_with_Ksep = 0
    Rpnew_Ksep_1 = 0
    pb_atma_tab = 0
    Rpnew = 0
    GasSol = 1  # передается в функцию mod_after_separation как аргумент
    GasGoesIntoSolution = 1  # проинициализирован в u7_types

    Rs, Bo = calc_rs_bo_m3m3(Pintake_, Tintake_, Qliq_, Fw_, class_pvt, z_class, pseudo_class)

    Rpnew_with_Ksep = rp_m3m3 - (rp_m3m3 - Rs) * Ksep
    Rpnew_Ksep_1 = rp_m3m3 - (rp_m3m3 - Rs)

    Delta = (pb_atma - 1) / n
    i = 0
    Tintake_tres_C = 101.5
    func_array_1 = [[], []]
    func_array_2 = [[], []]
    while i <= 10:
        pb_atma_tab = 1 + Delta * i
        rsb_m3m3_tab, Bo_m3m3_tab = calc_rs_bo_m3m3(pb_atma_tab, Tintake_tres_C, Qliq_, Fw_, class_pvt, z_class, pseudo_class)
        # добавляем элементы в два массива. Они будут представлять из себя два "графика"
        func_array_1[x_gr].append(rsb_m3m3_tab)  # будет представлять себя массив точек X
        func_array_1[y_gr].append(pb_atma_tab)  # будет представлять себя массив точек Y
        func_array_2[x_gr].append(rsb_m3m3_tab)
        func_array_2[y_gr].append(Bo_m3m3_tab)
        i = i + 1

    if GasSol == GasGoesIntoSolution:
        Rpnew = Rpnew_with_Ksep
    else:
        Rpnew = Rpnew_Ksep_1

    if Rpnew < class_pvt.rsb_m3m3:
        class_pvt.pb_atma, class_pvt.bob_m3m3 = getPoint(Rpnew, i, func_array_1, func_array_2)
        class_pvt.rsb_m3m3 = Rpnew

    class_pvt.rp_m3m3 = Rpnew_with_Ksep

    class_pvt.calc(Pintake_, Tintake_, Qliq_, Fw_, z_class, pseudo_class)

# возвращаем в "исходное состояние"
    class_pvt.rp_m3m3 = rp_m3m3_r
    class_pvt.rsb_m3m3 = rsb_m3m3_r
    class_pvt.pb_atma = pb_atma_r
    class_pvt.bob_m3m3 = bob_m3m3_r

    return 0

def mod_separation_pipe(q_liq_, fw_, Length_, Pcalc_, Calc_along_flow_, class_pvt, Theta_deg, Dtub_, Hydr_corr_, Tcalc_, Tother_):
    # функция модификации свойств нефти после сепарации
    # удаление части газа меняет свойства нефти - причем добавление газа свойства не трогает
    # на входе условия при которых проходила сепарация
    z_class = z_factor_2015_kareem()
    pseudo_class = pseudo_standing()
    n = 10
    #  промежуточные переменные для сохранения значений этих параметров
    #  эти параметры должны возвращаться с теми значениями, с которыми они пришли
    rp_m3m3_r = class_pvt.rp_m3m3
    rsb_m3m3_r = class_pvt.rsb_m3m3
    bob_m3m3_r = class_pvt.bob_m3m3
    pb_atma_r = class_pvt.pb_atma
    rp_m3m3 = class_pvt.rp_m3m3

    class_pvt.rsb_m3m3 = class_pvt.rp_m3m3 # rsb_m3m3 должен быть равен rp_m3m3 вне зависимости какое изначально он значение имел до этого

    pb_atma = class_pvt.pb_atma
    Ksep = class_pvt.ksep_fr
    Rs = 0
    Bo = 0
    Rpnew_with_Ksep = 0
    Rpnew_Ksep_1 = 0
    pb_atma_tab = 0
    Rpnew = 0
    GasSol = 1  # передается в функцию mod_after_separation как аргумент
    GasGoesIntoSolution = 1  # проинициализирован в u7_types

    Rs, Bo = calc_rs_bo_m3m3(class_pvt.pksep_atma, class_pvt.tksep_C, q_liq_, fw_, class_pvt, z_class, pseudo_class)

    Rpnew_with_Ksep = rp_m3m3 - (rp_m3m3 - Rs) * Ksep
    Rpnew_Ksep_1 = rp_m3m3 - (rp_m3m3 - Rs)

    Delta = (pb_atma - 1) / n
    i = 0
    Tintake_tres_C = 101.5
    func_array_1 = [[], []]
    func_array_2 = [[], []]
    while i <= n:
        pb_atma_tab = 1 + Delta * i
        rsb_m3m3_tab, Bo_m3m3_tab = calc_rs_bo_m3m3(pb_atma_tab, Tintake_tres_C, q_liq_, fw_, class_pvt, z_class, pseudo_class)
        # добавляем элементы в два массива. Они будут представлять из себя два "графика"
        func_array_1[x_gr].append(rsb_m3m3_tab)  # будет представлять себя массив точек X
        func_array_1[y_gr].append(pb_atma_tab)  # будет представлять себя массив точек Y
        func_array_2[x_gr].append(rsb_m3m3_tab)
        func_array_2[y_gr].append(Bo_m3m3_tab)
        i = i + 1

    if GasSol == GasGoesIntoSolution:
        Rpnew = Rpnew_with_Ksep
    else:
        Rpnew = Rpnew_Ksep_1

    if Rpnew < class_pvt.rsb_m3m3:
        class_pvt.pb_atma, class_pvt.bob_m3m3 = getPoint(Rpnew, i, func_array_1, func_array_2)
        class_pvt.rsb_m3m3 = Rpnew

    class_pvt.rp_m3m3 = Rpnew_with_Ksep

    pressure = cpipe.pipe_atma_pvtcls(q_liq_, fw_, Length_, Pcalc_, Calc_along_flow_, class_pvt, Theta_deg, Dtub_,
                                          Hydr_corr_, Tcalc_, Tother_)

    # возвращаем в "исходное состояние"
    class_pvt.rp_m3m3 = rp_m3m3_r
    class_pvt.rsb_m3m3 = rsb_m3m3_r
    class_pvt.pb_atma = pb_atma_r
    class_pvt.bob_m3m3 = bob_m3m3_r

    #    PVTdic['rp_m3m3'] = rp_m3m3_r
    #    PVTdic['rsb_m3m3'] = rsb_m3m3_r
    #    PVTdic['pb_atma'] = pb_atma_r
    #    PVTdic['bob_m3m3'] = bob_m3m3_r

    return pressure





z_class_select = z_factor_2015_kareem()
pseudo_class_select = pseudo_standing()

