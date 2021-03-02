import math_ext as m
import calc_func
import const

#import common.myLogger

def gas_fvf(temperature, pump_presure, z):
# Расчет объемного коэффициента газа при известном коэффиенте сжимаемости газа
    if temperature != temperature or (pump_presure == 0) or pump_presure != pump_presure or z != z: return None
    return 0.00034722 * temperature * z / pump_presure

def pseudo_pressure(gg):
    if ((4.9 - 0.4 * gg) == 0): return None
    return 4.9 - 0.4 * gg

def pseudo_temperature(gg):
    return 95 + 171 * gg

def pseudo_temperature_standing(gg):
    return 93.3 + 180 * gg - 6.94 * m.pow_(gg, 2)

def pseudo_pressure_standing(gg):
    return 4.6 + 0.1 * gg - 0.258 * m.pow_(gg, 2)

def dead_oil_viscosity_beggs_robin(temperature_k, gamma_oil):
    if (gamma_oil is None or gamma_oil == 0): return None
    x = m.pow_((1.8 * temperature_k - 460), -1.163) * m.exp_(13.108 - 6.591 / gamma_oil)
    return m.pow_(10, x) - 1

def saturated_oil_viscosity_beggs_r(gor_pb_m3m3, dead_oil_viscosity):
    if gor_pb_m3m3 != gor_pb_m3m3 or dead_oil_viscosity != dead_oil_viscosity:
        return None
    a = 10.715 * m.pow_(5.615 * gor_pb_m3m3 + 100, -0.515)
    b = 5.44 * m.pow_(5.615 * gor_pb_m3m3 + 150, -0.338)
    return a * m.pow_(dead_oil_viscosity, b)

def water_fvf(pressure_mpa, temperature_k):
# Расчет объемного коэффициента воды
    if pressure_mpa != pressure_mpa or  temperature_k != temperature_k:
        return None
    f = 1.8 * float(temperature_k) - 460
    psi = float(pressure_mpa) * 145.04
    dvwp = -1.95301 * m.pow_(10, -9) * psi * f - 1.72834 * m.pow_(10, -13) * m.pow_(psi, 2) * f - 3.58922 * m.pow_(10, -7) * psi - 2.25341 * m.pow_(10, -10) * m.pow_(psi, 2)
    dvwt = -1.0001 * m.pow_(10, -2) + 1.33391 * m.pow_(10, -4) * f + 5.50654 * m.pow_(10, -7) * m.pow_(f, 2)
    return (1 + dvwp) * (1 + dvwt)

def water_viscosity(pressure_mpa, temperature_k, salinity_mg_liter, gamma_water):
# Расчет вязкости воды
    if pressure_mpa != pressure_mpa or  temperature_k != temperature_k or gamma_water != gamma_water:
        return None
    if (gamma_water is None or gamma_water == 0): return None
    wptds = salinity_mg_liter / (10000 * gamma_water)
    a = 109.574 - 8.40564 * wptds + 0.313314 * m.pow_(wptds, 2) + 0.00872213 * m.pow_(wptds, 3)
    b = -1.12166 + 0.0263951 * wptds - 0.000679461 * m.pow_(wptds, 2) - 5.47119 * m.pow_(10, -5) * m.pow_(wptds, 3) + 1.55586 * m.pow_(10, -6) * m.pow_(wptds, 4)
    visc = a * m.pow_(1.8 * temperature_k - 460, b)
    psi = pressure_mpa * 145.04
    return visc * (0.9994 + 4.0295 * m.pow_(10, -5) * psi + 3.1062 * m.pow_(10, -9) * m.pow_(psi, 2))

def g_visc(t, p, z, gg):
# расчет вязкости газа
    if p != p or z != z or gg != gg or (t == 0): return None
    r = 1.8 * t
    mwg = 28.966 * gg
    gd = p * mwg / (z * t * 8.31)
    a = (9.379 + 0.01607 * mwg) * m.pow_(r, 1.5) / (209.2 + 19.26 * mwg + r)
    b = 3.448 + 986.4 / r + 0.01009 * mwg
    c = 2.447 - 0.2224 * b
    if (b is None or gd is None or c is None or m.pow_(gd, c) is None): return None
    return 0.0001 * a * m.exp_(b * m.pow_(gd, c))


def bubble_point_standing(rsb_m3m3, gamma_gas, temperature_k, gamma_oil):
    min_rsb = 1.8
    _bubblepoint_standing = 0
    yg = 0
    rsb_old = rsb_m3m3
    rsb_m3m3calc = rsb_m3m3;

    if (rsb_m3m3calc < min_rsb):
        rsb_m3m3calc = min_rsb;
    if (gamma_oil == 0): return None

    yg = 1.225 + 0.001648 * temperature_k - 1.769 / gamma_oil
    _bubblepoint_standing = 0.5197 * m.pow_(rsb_m3m3calc / gamma_gas, 0.83) * m.pow_(10, yg)

    if (rsb_old < min_rsb):
        _bubblepoint_standing = (_bubblepoint_standing - 0.1013) * rsb_old / min_rsb + 0.1013
    return _bubblepoint_standing

def fvf_saturated_oil_standing(rs_m3m3, gamma_gas, temperature_k, gamma_oil):
    if (gamma_oil == 0): return None
    f =5.615 * rs_m3m3 * m.pow_(gamma_gas / gamma_oil, 0.5)+ 2.25 * temperature_k - 575
    return 0.972 + 0.000147 * m.pow_(f, 1.175)

def compressibility_oil_vb(rs_m3m3, gamma_gas, temperature_k, gamma_oil, pressure_mpa):
    if (gamma_oil == 0 or pressure_mpa == 0): return None
    return (28.1 * rs_m3m3 + 30.6 * temperature_k - 1180 * gamma_gas + 1784 / gamma_oil - 10910)/ (100000 * pressure_mpa)

def oil_viscosity_vasquez_beggs(saturated_oil_viscosity, pressure_mpa, bp_pressure_mpa):
    c1 = 957
    c2 = 1.187
    c3 = -11.513
    c4 = -0.01302
    if (pressure_mpa is None or bp_pressure_mpa == 0): return None
    pow_gradle = c1 * m.pow_(pressure_mpa, c2) * m.exp_(c3 + c4 * pressure_mpa)
    return saturated_oil_viscosity * m.pow_(pressure_mpa / bp_pressure_mpa, pow_gradle)

def gor_standing(pressure_mpa, gamma_gas, temperature_k, gamma_oil):
    if (gamma_oil == 0): return None
    yg = 1.225 + 0.001648 * temperature_k - 1.769 / gamma_oil
    return gamma_gas * m.pow_(1.92 * pressure_mpa / m.pow_(10, yg), 1.204)

def zfactor(tpr, ppr):
    if (ppr is None or tpr is None or tpr == 0.86): return None
    a = 1.39 * m.pow_((tpr - 0.92), 0.5) - 0.36 * tpr - 0.101
    b = ppr * (0.62 - 0.23 * tpr) + m.pow_(ppr, 2) * (0.006 / (tpr - 0.86) - 0.037) + 0.32 * m.pow_(ppr, 6) / m.exp_(20.723 * (tpr - 1))
    if (b is None): return None
    c = 0.132 - 0.32 * m.log_(tpr) / m.log_(10)
    d = m.exp_(0.715 - 1.128 * tpr + 0.42 * m.pow_(tpr, 2))
    if (a is None or m.exp_(-b) is None or c is None or m.pow_(ppr, d) is None): return None
    result = a + (1 - a) * m.exp_(-b) + c * m.pow_(ppr, d)
    return result

def dead_oil_viscosity_standing(temperature_k, gamma_oil):
    if (gamma_oil is None or temperature_k is None or  gamma_oil == 0 or (141.5 / gamma_oil - 131.5) == 0 or (1.8 * temperature_k - 260) == 0):
        return None
    result = (0.32 + 1.8 * m.pow_(10, 7) / m.pow_(141.5 / gamma_oil - 131.5, 4.53)) * m.pow_(360 / (1.8 * temperature_k - 260), m.pow_(10, (0.43 + 8.33 / (141.5 / gamma_oil - 131.5))))
    return result

def fvf_mccainsi(rs_m3m3, gamma_gas, sto_density_kg_m3, reservoir_oil_density_kg_m3):
    if (reservoir_oil_density_kg_m3 == 0): return None
    return (sto_density_kg_m3 + 1.22117 * rs_m3m3 * gamma_gas) / reservoir_oil_density_kg_m3

def oil_viscosity_standing(rs_m3m3, dead_oil_viscosity, pressure_mpa, bubblepoint_pressure_mpa):
    a = 5.6148 * rs_m3m3 * (0.1235 * m.pow_(10, -5) * rs_m3m3 - 0.00074)
    b = 0.68 / m.pow_(10, 0.000484 * rs_m3m3) + 0.25 / m.pow_(10, 0.006176 * rs_m3m3) + 0.062 / m.pow_(10, 0.021 * rs_m3m3)
    _oil_viscosity_standing = m.pow_(10, a) * m.pow_(dead_oil_viscosity, b)
    if (bubblepoint_pressure_mpa < pressure_mpa):
        _oil_viscosity_standing = _oil_viscosity_standing + 0.14504 * (pressure_mpa - bubblepoint_pressure_mpa) * (0.024 * m.pow_(_oil_viscosity_standing, 1.6) + 0.038 * m.pow_(_oil_viscosity_standing, 0.56))
    return _oil_viscosity_standing

def bubble_point_valko_mccainsi(rsb_m3m3, gamma_gas, temperature_k, gamma_oil):
    min_rsb = 1.8
    max_rsb = 800
    rsb_old = 0
    rsb_m3m3calc = 0
    _bubblepoint_valko_mccainsi = 0
    z1 = 0
    z2 = 0
    z3 = 0
    z4 = 0
    z = 0
    api = 0
    lnpb = 0
    if (rsb_m3m3 is None or gamma_gas is None or temperature_k is None or gamma_oil is None or gamma_oil == 0): return None
    rsb_m3m3calc = rsb_m3m3
    rsb_old = rsb_m3m3calc
    rsb_m3m3calc = rsb_m3m3

    if (rsb_m3m3calc < min_rsb): rsb_m3m3calc = min_rsb
    if (rsb_m3m3calc > max_rsb): rsb_m3m3calc = max_rsb;

    api = 141.5 / gamma_oil - 131.5
    z1 = -4.814074834 + 0.7480913 * m.log_(rsb_m3m3calc) + 0.1743556 * m.pow_(m.log_(rsb_m3m3calc), 2) - 0.0206 * m.pow_(m.log_(rsb_m3m3calc), 3)
    z2 = 1.27 - 0.0449 * api + 4.36 * m.pow_(10, -4) * m.pow_(api, 2) - 4.76 * m.pow_(10, -6) * m.pow_(api, 3)
    z3 =4.51 - 10.84 * gamma_gas + 8.39 * m.pow_(gamma_gas, 2) - 2.34 * m.pow_(gamma_gas, 3)
    z4 = -7.2254661 + 0.043155 * temperature_k - 8.5548 * m.pow_(10, -5) * m.pow_(temperature_k, 2) + 6.00696 * m.pow_(10, -8) * m.pow_(temperature_k, 3)
    z = z1 + z2 + z3 + z4
    lnpb = 2.498006 + 0.713 * z + 0.0075 * m.pow_(z, 2)
    _bubblepoint_valko_mccainsi = m.pow_(2.718282, lnpb)

    if (rsb_old < min_rsb): _bubblepoint_valko_mccainsi = (_bubblepoint_valko_mccainsi - 0.1013) * rsb_old / min_rsb + 0.1013

    if (rsb_old > max_rsb): _bubblepoint_valko_mccainsi = (_bubblepoint_valko_mccainsi - 0.1013) * rsb_old / max_rsb + 0.1013
    return _bubblepoint_valko_mccainsi

def gor_velardesi(pressure_mpa, bubblepoint_pressure_mpa, gamma_gas, temperature_k, gamma_oil, rsb_m3_m3):
    maxrs = 800
    api =  gor_velardesi =  pr =  a_0 =  a_1 =  a_2 =  a_3 =  a_4 =  b_0 =  b_1 =  b_2 =  b_3 =  b_4 = 0
    c_0 =  c_1 =  c_2 =  c_3 =  c_4 =  a1 =  a2 =  a3 =  rsr = 0

    if (gamma_oil == 0): return None
    api = 141.5 / gamma_oil - 131.5
    if (bubblepoint_pressure_mpa > bubble_point_valko_mccainsi(maxrs, gamma_gas, temperature_k, gamma_oil)):
        if (pressure_mpa < bubblepoint_pressure_mpa):
            gor_velardesi = (rsb_m3_m3) * (pressure_mpa / bubblepoint_pressure_mpa)
        else:
            gor_velardesi = rsb_m3_m3
        return gor_velardesi


    if (bubblepoint_pressure_mpa > 0):
        pr = (pressure_mpa - 0.101) / (bubblepoint_pressure_mpa)
    else:
        pr = 0

    if (pr <= 0): return 0

    if (pr >= 1):
        gor_velardesi = rsb_m3_m3;

    if (pr < 1):
        a_0 = 1.8653 * m.pow_(10, -4)
        a_1 = 1.672608
        a_2 = 0.92987
        a_3 = 0.247235
        a_4 = 1.056052
        a1 = a_0 * m.pow_(gamma_gas, a_1) * m.pow_(api, a_2) * m.pow_(1.8 * temperature_k - 460, a_3) * m.pow_(bubblepoint_pressure_mpa, a_4)
        b_0 = 0.1004
        b_1 = -1.00475
        b_2 = 0.337711
        b_3 = 0.132795
        b_4 = 0.302065
        a2 = b_0 * m.pow_(gamma_gas, b_1) * m.pow_(api, b_2) * m.pow_(1.8 * temperature_k - 460, b_3) * m.pow_(bubblepoint_pressure_mpa, b_4)
        c_0 = 0.9167
        c_1 = -1.48548
        c_2 = -0.164741
        c_3 = -0.09133
        c_4 = 0.047094
        a3 = c_0 * m.pow_(gamma_gas, c_1) * m.pow_(api, c_2) * m.pow_(1.8 * temperature_k - 460, c_3) * m.pow_(bubblepoint_pressure_mpa, c_4)
        rsr = a1 * m.pow_(pr, a2) + (1 - a1) * m.pow_(pr, a3)
        gor_velardesi = rsr * rsb_m3_m3
    return gor_velardesi

def density_mccainsi(pressure_mpa, gamma_gas, temperature_k, gamma_oil, rs_m3_m3, bp_pressure_mpa, compressibility):
    maxiter = 1000
    rs_m3_m3calc = ropo = pm = pmmo =epsilon = a0 = a1 = a2 = a3 = a4 = a5 = 0
    _density_mccainsi = dpt = roa = dpp = pbs = bp_pressure_mpacalc = 0

    rs_m3_m3calc = rs_m3_m3
    bp_pressure_mpacalc = bp_pressure_mpa

    if (rs_m3_m3calc > 800):
        rs_m3_m3calc = 800
        bp_pressure_mpacalc = bubble_point_valko_mccainsi(rs_m3_m3calc, gamma_gas, temperature_k, gamma_oil)

    ropo = 845.8 - 0.9 * rs_m3_m3calc
    pm = ropo
    pmmo = 0
    epsilon = 0.000001
    i = 0
    counter = 0

    if (pmmo is None or pm is None): return None

    while (counter < maxiter and m.abs_(pmmo - pm) > epsilon):
        i = i + 1
        pmmo = pm
        a0 = -799.21
        a1 = 1361.8
        a2 = -3.70373
        a3 = 0.003
        a4 = 2.98914
        a5 = -0.00223
        roa = a0 + a1 * gamma_gas + a2 * gamma_gas * ropo + a3 * gamma_gas * m.pow_(ropo, 2) + a4 * ropo + a5 * m.pow_(ropo, 2)
        ropo = (rs_m3_m3calc * gamma_gas + 818.81 * gamma_oil) / (0.81881 + rs_m3_m3calc * gamma_gas / roa)
        pm = ropo
        counter = counter + 1


    if (pressure_mpa <= bp_pressure_mpacalc):
        dpp = (0.167 + 16.181 * m.pow_(10, (-0.00265 * pm))) * (2.32328 * pressure_mpa) - 0.16 * (0.299 + 263 * m.pow_(10, (-0.00376 * pm))) * m.pow_(0.14503774 * pressure_mpa, 2)
        pbs = pm + dpp
        dpt = (0.04837 + 337.094 * m.pow_(pbs, -0.951)) * m.pow_(1.8 * temperature_k - 520, 0.938) - (0.346 - 0.3732 * m.pow_(10, -0.001 * pbs)) * m.pow_(1.8 * temperature_k - 520, 0.475)
        pm = pbs - dpt
        _density_mccainsi = pm
    else:
        dpp = (0.167 + 16.181 * m.pow_(10, -0.00265 * pm)) * (2.32328 * bp_pressure_mpacalc) - 0.16 * (0.299 + 263 * m.pow_(10, -0.00376 * pm)) * m.pow_(0.14503774 * bp_pressure_mpacalc, 2)
        pbs = pm + dpp
        dpt =(0.04837 + 337.094 * m.pow_(pbs, -0.951)) * m.pow_(1.8 * temperature_k - 520, 0.938) - (0.346 - 0.3732 * m.pow_(10, -0.001 * pbs)) * m.pow_(1.8 * temperature_k - 520, 0.475)
        pm = pbs - dpt
        _density_mccainsi = pm * m.exp_(compressibility *(pressure_mpa - bp_pressure_mpacalc))
    return _density_mccainsi

def unf_calc_Sal_BwSC_ppm(BwSC):
# функция для оценки солености воды по объемному коэффициенту (получена как обратная к unf_calc_BwSC_d)
    salinity = m.sqrt_(624.711071129603 * BwSC / 0.0160185 - 20192.9595437054) - 137.000074965329
    return salinity * 10000

def Q_fluid_calc(qliq_sm3day, waterСut, RoilVol, RwatVol, rp_m3m3, qgas_free_sm3day, RgasVolOil, RgasVol):
    # Расчет дебитов нефти (qOil), воды (qWat), газа (qGas)
    #qliq_sm3day      - дебит жидкости
    #waterСut         - Обводненность, доля не процент
    #RoilVol          - объемный коэффициент нефти при рабочих условиях
    #RwatVol          - объемный коэффициент воды
    #rp_m3m3          - газовый фактор добычной (приведенный к стандартным условиям)
    #qgas_free_sm3day - свободный газ в потоке
    #RgasVolOil       - расчетное значение газосодержания в нефти при текущих условиях
    #RgasVol          - объемный коэффициент газа при известном коэффициенте сжимаемости газа

    if (qliq_sm3day == None or waterСut == None or RoilVol == None):
        q_oil_rc_m3day = None
    else:
        q_oil_rc_m3day = qliq_sm3day * (1 - waterСut) * RoilVol

    if (q_oil_rc_m3day == None or RwatVol == None):
        q_wat_rc_m3day = None
    else:
        q_wat_rc_m3day = qliq_sm3day * waterСut * RwatVol

    if (q_oil_rc_m3day == None or qgas_free_sm3day == None or RgasVolOil == None or RgasVol == None):
        q_gas_rc_m3day = None
    else:
        q_gas_rc_m3day = (qliq_sm3day * (1 - waterСut) * rp_m3m3 + qgas_free_sm3day - RgasVolOil * qliq_sm3day * (1 - waterСut)) * RgasVol
    return (q_oil_rc_m3day, q_wat_rc_m3day, q_gas_rc_m3day)

def getoil_API(gamma_oil):
    return 141.5 / gamma_oil - 131.5

def calc_ST(p_atma, t_C, water_сut_share, aPVTdic):
    #water_сut_share     - Обводненность, доля не процент
    ST68 = None
    ST100 = None
    STw74 = None
    STw280 = None
    Tst = None
    Tstw = None
    STo = None
    STw = None
    ST = None
    t_F = None
    P_psia = None
    P_MPa = None
    gamma_oil = None
    if 'gamma_oil' in aPVTdic:
        gamma_oil = aPVTdic['gamma_oil']
    if gamma_oil != gamma_oil or gamma_oil == None or gamma_oil == 0:
        return None
    # calculate surface tension according Baker Sverdloff correlation
    # VB2PY (UntranslatedCode) On Error GoTo err1
    #Расчет коэффициента поверхностного натяжения газ-нефть
    t_F = t_C * 1.8 + 32
    P_psia = p_atma / 0.068046
    P_MPa = p_atma / 10
    oil_API = getoil_API(gamma_oil)
    ST68 = 39 - 0.2571 * oil_API
    ST100 = 37.5 - 0.2571 * oil_API
    if t_F < 68:
        STo = ST68
    else:
        Tst = t_F
        if t_F > 100:
            Tst = 100
        STo = (68 -(((Tst - 68) * (ST68 - ST100))/ 32)) * (1 - (0.024 * (P_psia) ** 0.45))
        if STo < 0:
            STo = ST68
            # todo rnt20190312 надо будет исправить когда то
    #Расчет коэффициента поверхностного натяжения газ-вода  (два способа)
    # Первый способ - выключен, т.к. второй способ перекрывает первый
    # STw74 = (75 - (1.108 * (P_psia) ** 0.349))
    # STw280 = (53 - (0.1048 * (P_psia) ** 0.637))
    # if t_F < 74:
    #     STw = STw74
    # else:
    #     Tstw = t_F
    #     if t_F > 280:
    #         Tstw = 280
    #     STw = STw74 - (((Tstw - 74) * (STw74 - STw280)) / 206)
    # далее второй способ
    STw = 10 ** (-(1.19 + 0.01 * P_MPa)) * 1000
    # Расчет коэффициента поверхностного натяжения газ-жидкость
    ST = (STw * water_сut_share) + STo * (1 - water_сut_share)
    aPVTdic.update({'ST_oilgas_dyncm': STo})
    aPVTdic.update({'ST_watgas_dyncm': STw})
    aPVTdic.update({'ST_liqgas_dyncm': ST})
    return aPVTdic

def calc_pvt(p, tIn, rSb, gammaO, gammaG, yukosStandard, pRb, bRo, s, gammaW):
    if (pRb is None or pRb == 0): pRb = -1
    cPMpa = 0.1013
    rhoRef = 1000
    rhoAir = 1.2217
    pseudo_pressure_ = 0
    pseudo_temperature_ = 0
    compressibility_at_bubble_point_pressure_ = 0
    rho_o = 0
    b_o_sat = 0
    saturated_oil_viscosity_beggs_robinson_ = 0
    dead_oil_viscosity_ = 0
    tPr = 0
    pPr = 0
    bubble_point_standing_ = 0
    gor_velardesi_ = 0
    b_oi = 0
    oil_viscosity_vasquez_beggs_ = 0
    zi = 0
    b_gi = 0
    mu_gi = 0
    b_wi = 0
    mu_wi = 0
    rho_o_sat = 0
    p_fact = 0
    p_offs = 0
    b_fact = 0
    oil_pressure_in_MPa_ = 0
    p_rbcalc = 0
    t = tIn

    if (yukosStandard > 0):
        pseudo_pressure_ = pseudo_pressure(gammaG)
        pseudo_temperature_ = pseudo_temperature(gammaG)
    else:
        pseudo_pressure_ = pseudo_pressure_standing(gammaG)
        pseudo_temperature_ = pseudo_temperature_standing(gammaG)

    oil_pressure_in_MPa_ = p * cPMpa
    ##convert user specified bubblepoint pRb - pressure давление насыщения  (калибровочное значение)
    p_rbcalc = pRb * cPMpa
    ##for saturated oil calibration is applied by application of factor p_fact to input pressure
    ##for undersaturated - by shifting according to p_offs
    ##calculate PVT properties
    ##calculate water properties at current pressure and temperature
    b_wi = water_fvf(oil_pressure_in_MPa_, t)
    mu_wi = water_viscosity(oil_pressure_in_MPa_, t, s, gammaW)

    if (yukosStandard == 0):
    ##Not Yukos standard set of correlations
    ##Gas properties
        pPr = oil_pressure_in_MPa_ / pseudo_pressure_
        tPr = t / pseudo_temperature_
#       zi = calc_func.z_factor_dran_chuk(tPr, pPr) - вызов до ввода классов, были значения вместо словаря
        z_class = calc_func.z_factor_dran_chuk()
        args_dic = {}
        args_dic['t_pseudo'] = tPr
        args_dic['p_pseudo'] = pPr
        zi = z_class.z_factor(args_dic)
        b_gi = gas_fvf(t, oil_pressure_in_MPa_, zi)
        mu_gi = g_visc(t, oil_pressure_in_MPa_, zi, gammaG)
    ##dead oil viscosity
        dead_oil_viscosity_ = dead_oil_viscosity_beggs_robin(t, gammaO)
    ##saturated oil viscosity Beggs Robinson
        saturated_oil_viscosity_beggs_robinson_ = saturated_oil_viscosity_beggs_r(rSb, dead_oil_viscosity_)
    ##Standing
        bubble_point_standing_ = bubble_point_standing(rSb, gammaG, t, gammaO)
    ##Calculate bubble point correction factor
        if (p_rbcalc > 0):
    ##user specified
            p_fact = bubble_point_standing_ / p_rbcalc
            p_offs = bubble_point_standing_ - p_rbcalc
        else:
    ##not specified, use from correlations
            p_fact = 1
            p_offs = 0

    ##Calculate oil formation volume factor correction factor
        if (bRo > 0):
    ##user specified
            b_o_sat = fvf_saturated_oil_standing(rSb, gammaG, t, gammaO)
            b_fact = (bRo - 1) / (b_o_sat - 1)
        else:
    ##not specified, use from correlations
            b_fact = 1

        if (oil_pressure_in_MPa_ > (bubble_point_standing_ / p_fact)):
    ##undersaturated oil
    ##apply correction to undersaturated oil
            oil_pressure_in_MPa_ = oil_pressure_in_MPa_ + p_offs
            gor_velardesi_ = rSb
    ##Standing
            b_o_sat = b_fact * (fvf_saturated_oil_standing(gor_velardesi_, gammaG, t, gammaO) - 1) + 1 ##it is assumed that at pressure 1 atm bo = 1
    ##calculate compressibility at bubble point pressure
            compressibility_at_bubble_point_pressure_ = compressibility_oil_vb(rSb, gammaG, t, gammaO, oil_pressure_in_MPa_)
            b_oi = b_o_sat * m.exp_(compressibility_at_bubble_point_pressure_ * (bubble_point_standing_ - oil_pressure_in_MPa_))
    ##VesquezBeggs
            oil_viscosity_vasquez_beggs_ = oil_viscosity_vasquez_beggs(saturated_oil_viscosity_beggs_robinson_, oil_pressure_in_MPa_, bubble_point_standing_)
        else:
    ##saturated oil
    ##apply correction to saturated oil
            oil_pressure_in_MPa_ = oil_pressure_in_MPa_ * p_fact
    ##Standing
            gor_velardesi_ = gor_standing(oil_pressure_in_MPa_, gammaG, t, gammaO)
    ##Standing
            b_oi = b_fact * (fvf_saturated_oil_standing(gor_velardesi_, gammaG, t, gammaO) - 1) + 1 ##it is assumed that at pressure 1 atm bo = 1
    ##Beggs Robinson
            oil_viscosity_vasquez_beggs_ = saturated_oil_viscosity_beggs_r(gor_velardesi_, dead_oil_viscosity_)

    if (yukosStandard == 1):
    ##Yukos standard set of correlations
    ##gas properties
        pPr = oil_pressure_in_MPa_ / pseudo_pressure_
        tPr = t / pseudo_temperature_
    ##Debug, remove xxx
    ##Zi = ZFactorDranchuk(T_pr, p_pr)
        zi = zfactor(tPr, pPr)
        b_gi = gas_fvf(t, oil_pressure_in_MPa_, zi)
        mu_gi = g_visc(t, oil_pressure_in_MPa_, zi, gammaG)
    ##dead oil viscosity
        dead_oil_viscosity_ = dead_oil_viscosity_standing(t, gammaO)
    ##saturated oil viscosity Beggs Robinson
        saturated_oil_viscosity_beggs_robinson_ = saturated_oil_viscosity_beggs_r(rSb, dead_oil_viscosity_)
        bubble_point_standing_ = bubble_point_valko_mccainsi(rSb, gammaG, t, gammaO)
    ##Standing - debug, remove xxx
    ##p_bi = Bubblepoint_Standing(r_sb, gamma_g, t, gamma_o)
    ##Calculate bubble point correction factor
        if (p_rbcalc > 0):
    ##user specifie
            p_fact = bubble_point_standing_ / p_rbcalc
            p_offs = bubble_point_standing_ - p_rbcalc
        else:
    ##not specified, use from correlations
            p_fact = 1
            p_offs = 0

        if (oil_pressure_in_MPa_ > (bubble_point_standing_ / p_fact)):
    ##undersaturated oil
    ##apply correction to undersaturated oil
            oil_pressure_in_MPa_ = oil_pressure_in_MPa_ + p_offs
        else:
    ##apply correction to saturated oil
            oil_pressure_in_MPa_ = oil_pressure_in_MPa_ * p_fact

    ##Calculate oil formation volume factor correction factor
        if (bRo > 0):
    ##user specified
            rho_o_sat = density_mccainsi(bubble_point_standing_, gammaG, t, gammaO, rSb, bubble_point_standing_, compressibility_at_bubble_point_pressure_)
            b_o_sat = fvf_mccainsi(rSb, gammaG, gammaO * rhoRef, rho_o_sat)
            b_fact = (bRo - 1) / (b_o_sat - 1)
        else:
    ##not specified, use from correlations
            b_fact = 1

    ##Debug, uncomment xxx
        gor_velardesi_ = gor_velardesi(oil_pressure_in_MPa_, bubble_point_standing_, gammaG, t, gammaO, rSb)
    ##calculate compressibility at bubble point pressure
        compressibility_at_bubble_point_pressure_ = compressibility_oil_vb(rSb, gammaG, t, gammaO, oil_pressure_in_MPa_)

        if (oil_pressure_in_MPa_ > bubble_point_standing_):
    ##undersaturated oil
    ##Debug, remove xxx
    ##r_si = r_sb
    ##apply correction to undersaturated oil
            rho_o_sat = density_mccainsi(bubble_point_standing_, gammaG, t, gammaO, rSb, bubble_point_standing_, compressibility_at_bubble_point_pressure_)
            b_o_sat = fvf_mccainsi(rSb, gammaG, gammaO * rhoRef, rho_o_sat)
            b_o_sat = b_fact * (b_o_sat - 1) + 1
    ##it is assumed that at pressure 1 atm bo = 1
            b_oi = b_o_sat * m.exp_(compressibility_at_bubble_point_pressure_ * (bubble_point_standing_ - oil_pressure_in_MPa_))
        else:
    ##Debug, remove xxx
    ##r_si = GOR_Standing(p_mpa, gamma_g, t, gamma_o)
    ##apply correction to saturated oil
            rho_o = density_mccainsi(oil_pressure_in_MPa_, gammaG, t, gammaO, gor_velardesi_, bubble_point_standing_, compressibility_at_bubble_point_pressure_)
            b_oi = b_fact * (fvf_mccainsi(gor_velardesi_, gammaG, gammaO * rhoRef, rho_o) - 1) + 1 ##it is assumed that at pressure 1 atm bo = 1

        if (rSb < 350):
    ##Calculate oil viscosity acoording to Standing
            oil_viscosity_vasquez_beggs_ = oil_viscosity_standing(gor_velardesi_, dead_oil_viscosity_, oil_pressure_in_MPa_, bubble_point_standing_)
        else:
    ##Calculate according to BegsRobinson(saturated)and VasquezBegs(undersaturated)
            if (oil_pressure_in_MPa_ > bubble_point_standing_):
    ##undersaturated oil
                oil_viscosity_vasquez_beggs_ = oil_viscosity_vasquez_beggs(saturated_oil_viscosity_beggs_robinson_, oil_pressure_in_MPa_, bubble_point_standing_)
            else:
    ##saturated oil
    ##Beggs Robinson
                oil_viscosity_vasquez_beggs_ = saturated_oil_viscosity_beggs_r(gor_velardesi_, dead_oil_viscosity_)

    if (yukosStandard == 2):
        ##Debug mode.Linear Rs and bo vs P, p_rb should be specified.
        ##gas properties
        ##ideal gas
        zi = 1
        ##debug
        ##p_pr = p_mpa / p_pc
        ##T_pr = t / T_pc
        ##Zi = ZFactorDranchuk(T_pr, p_pr)
        b_gi = gas_fvf(t, oil_pressure_in_MPa_, zi)
        mu_gi = 0.0000000001
        ##Set to default.b_rb should be specified by user!
        p_fact = 1
        p_offs = 0
        bubble_point_standing_ = p_rbcalc
        if (oil_pressure_in_MPa_ > (bubble_point_standing_)):
            ##undersaturated oil
            gor_velardesi_ = rSb
        else:
            ##saturate
            gor_velardesi_ = oil_pressure_in_MPa_ / p_rbcalc * rSb
            ##r_si = GOR_Standing(p_mpa * p_fact, gamma_g, t, gamma_o)

        ##if b_o is not specified by the user then
        ##set b_o so, that oil density, recalculated with r_s would be equal to dead oil density
        if (bRo < 0):
            if (gammaO == 0):
                bRo = None
            else:
                b_oi = (1 + gor_velardesi_ * (gammaG * rhoAir) / (gammaO * rhoRef))
        else:
            if (oil_pressure_in_MPa_ > bubble_point_standing_):
                b_oi = bRo ##undersaturated oil
            else:
                b_oi = 1.0 + (bRo - 1.0) * (oil_pressure_in_MPa_ - cPMpa) / (bubble_point_standing_ - cPMpa)
        oil_viscosity_vasquez_beggs_ = 1
    ##Assign output variables
        # расчетное значение давления насыщения по корреляции
    Psat = bubble_point_standing_ / cPMpa / p_fact
        # расчетное значение газосодержания в нефти при текущих условиях
    RgasVolOil = gor_velardesi_
        # объемный коэффициент нефти при рабочих условиях
    RoilVol = b_oi
        # вязкость нефти при рабочих условиях
    VisOil = oil_viscosity_vasquez_beggs_
        # расчетное значение коэффициента сверхсжимаемости газа
    Zfactor = zi
        # объемный коэффициент газа при известном коэффициенте сжимаемости газа
    RgasVol = b_gi
        # вязкость газа при рабочих условиях
    VisGas = mu_gi
        # объемный коэффициент воды
    RwatVol = b_wi
        # вязкость воды
    VisWat = mu_wi
    return Psat, RgasVolOil, RoilVol, VisOil, Zfactor, RgasVol, VisGas, RwatVol, VisWat

def calc_pvt_vr(pksep_atma, tIn_C, Qliq, water_сut_share, PVT_CORRELATION, aPVTdic, z_factor_func, pseudo_crit_func):
# Расчет свойств флюидов UF7.14
# water_сut_share     - Обводненность, доля не процент
#    log.logger.debug(f'Module: "{__name__}"       Function: "{calc_pvt_vr.__name__}"       Current parameters: pksep_atma = {pksep_atma}, tIn_C = {tIn_C}, Qliq = {Qliq}, water_сut_share = {water_сut_share}')
    #common_vr.log.logger.info('Если требуется, можно включить запись информационных логов. Запись ведётся в свой журнал.')
    try:
        if aPVTdic is None or len(aPVTdic) == 0:
            return None
        if (aPVTdic['pb_atma'] is None or aPVTdic['pb_atma'] == 0):
            pb_atma = -1
        else:
            pb_atma = aPVTdic['pb_atma']
        rsb_m3m3 = aPVTdic['rsb_m3m3']
        gamma_oil = aPVTdic['gamma_oil']
        gamma_gas = aPVTdic['gamma_gas']
        gamma_wat = aPVTdic['gamma_wat']
        bob_m3m3 = aPVTdic['bob_m3m3']
        salinity_ppm = aPVTdic['salinity_ppm']
        tres_C = aPVTdic['tres_C']
        muob_cP = aPVTdic['muob_cP']
        rp_m3m3 = None
        if 'rp_m3m3' in aPVTdic:
            rp_m3m3 = aPVTdic['rp_m3m3']
        qgas_free_sm3day = None
        if 'qgas_free_sm3day' in aPVTdic:
            qgas_free_sm3day = aPVTdic['qgas_free_sm3day']
        cPMpa = 0.101325
        rhoRef = 1000
        rhoAir = 1.2217
        const_t_K_min = 273
        pseudo_pressure_ = 0
        pseudo_temperature_ = 0
        compressibility_at_bubble_point_pressure_ = 0
        rho_o = 0
        b_o_sat = 0
        saturated_oil_viscosity_beggs_robinson_ = 0
        dead_oil_viscosity_ = 0
        tPr = 0
        pPr = 0
        bubble_point_standing_ = 0
        rs_m3m3 = 0
        bo_m3m3 = 0
        mu_oil_cP = 0
        Zfactor = 0
        bg_m3m3 = 0
        mu_gas_cP = 0
        bw_m3m3 = 0
        mu_wat_cP = 0
        rho_o_sat = 0
        p_fact = 0
        p_offs = 0
        b_fact = 0
        oil_pressure_in_MPa_ = 0
        p_rbcalc = 0
        q_oil_rc_m3day = None
        q_wat_rc_m3day = None
        q_gas_rc_m3day = None

        mcalibr_cP = muob_cP

        t = tIn_C + const_t_K_min
        t_res_K = tres_C + const_t_K_min
        pseudo_temperature_, pseudo_pressure_ = pseudo_crit_func.press_temp(gamma_gas)

        oil_pressure_in_MPa_ = pksep_atma * cPMpa
        ##convert user specifi1ed bubblepoint pb_atma - pressure давление насыщения  (калибровочное значение)
        # 14.084175
        p_rbcalc = pb_atma * cPMpa
        ##for saturated oil calibration is applied by application of factor p_fact to input pressure
        ##for undersaturated - by shifting according to p_offs
        ##calculate PVT properties
        ##calculate water properties at current pressure and temperature
        bw_m3m3 = water_fvf(oil_pressure_in_MPa_, t)

        mu_wat_cP = water_viscosity(oil_pressure_in_MPa_, t, salinity_ppm, gamma_wat)

        if (PVT_CORRELATION == 0):
        ##Not Yukos standard set of correlations
        ##Gas properties
            pPr = oil_pressure_in_MPa_ / pseudo_pressure_
            tPr = t / pseudo_temperature_
            args_dic = {'t_pseudo': tPr, 'p_pseudo': pPr}
            Zfactor = z_factor_func.z_factor(args_dic)
            bg_m3m3 = gas_fvf(t, oil_pressure_in_MPa_, Zfactor)
            mu_gas_cP = g_visc(t, oil_pressure_in_MPa_, Zfactor, gamma_gas)
        ##dead oil viscosity
            dead_oil_viscosity_ = dead_oil_viscosity_beggs_robin(t, gamma_oil)
        ##saturated oil viscosity Beggs Robinson
            saturated_oil_viscosity_beggs_robinson_ = saturated_oil_viscosity_beggs_r(rsb_m3m3, dead_oil_viscosity_)
        ##Standing
            bubble_point_standing_ = bubble_point_standing(rsb_m3m3, gamma_gas, t_res_K, gamma_oil)
        ##Calculate bubble point correction factor
            if (p_rbcalc > 0):
        ##user specified
                p_fact = bubble_point_standing_ / p_rbcalc
            else:
        ##not specified, use from correlations
                p_fact = 1

        ##Calculate oil formation volume factor correction factor
            if (bob_m3m3 > 0):
        ##user specified
                b_o_sat = fvf_saturated_oil_standing(rsb_m3m3, gamma_gas, t_res_K, gamma_oil)
                b_fact = (bob_m3m3 - 1) / (b_o_sat - 1)
            else:
        ##not specified, use from correlations
                b_fact = 1

            if mcalibr_cP > 0:
                mu_fact = mcalibr_cP / saturated_oil_viscosity_beggs_robinson_
            else:
                mu_fact = 1

            oil_pressure_in_MPa_ = oil_pressure_in_MPa_ * p_fact
            p_bi = bubble_point_standing(rsb_m3m3, gamma_gas, t, gamma_oil)

            if (oil_pressure_in_MPa_ > p_bi): # было (oil_pressure_in_MPa_ > (bubble_point_standing_ / p_fact))
        ##undersaturated oil
        ##apply correction to undersaturated oil
                rs_m3m3 = rsb_m3m3
        ##Standing
                b_o_sat = b_fact * (fvf_saturated_oil_standing(rs_m3m3, gamma_gas, t, gamma_oil) - 1) + 1 ##it is assumed that at pressure 1 atm bo = 1
        ##calculate compressibility at bubble point pressure
                compressibility_at_bubble_point_pressure_ = compressibility_oil_vb(rsb_m3m3, gamma_gas, t, gamma_oil, oil_pressure_in_MPa_)
                bo_m3m3 = b_o_sat * m.exp_(compressibility_at_bubble_point_pressure_ * (bubble_point_standing_ - oil_pressure_in_MPa_))
        ##VesquezBeggs
                mu_oil_cP = mu_fact * oil_viscosity_vasquez_beggs(saturated_oil_viscosity_beggs_robinson_, oil_pressure_in_MPa_, bubble_point_standing_)
            else:
        ##saturated oil
        ##apply correction to saturated oil
        ##Standing
                rs_m3m3 = gor_standing(oil_pressure_in_MPa_, gamma_gas, t, gamma_oil)
        ##Standing
                bo_m3m3 = b_fact * (fvf_saturated_oil_standing(rs_m3m3, gamma_gas, t, gamma_oil) - 1) + 1 ##it is assumed that at pressure 1 atm bo = 1
        ##Beggs Robinson
                mu_oil_cP = mu_fact * saturated_oil_viscosity_beggs_r(rs_m3m3, dead_oil_viscosity_)

        if (PVT_CORRELATION == 1):
        ##Yukos standard set of correlations
        ##gas properties
            pPr = oil_pressure_in_MPa_ / pseudo_pressure_
            tPr = t / pseudo_temperature_
        ##Debug, remove xxx
        ##Zi = ZFactorDranchuk(T_pr, p_pr)
            Zfactor = zfactor(tPr, pPr)
            bg_m3m3 = gas_fvf(t, oil_pressure_in_MPa_, Zfactor)
            mu_gas_cP = g_visc(t, oil_pressure_in_MPa_, Zfactor, gamma_gas)
        ##dead oil viscosity
            dead_oil_viscosity_ = dead_oil_viscosity_standing(t, gamma_oil)
        ##saturated oil viscosity Beggs Robinson
            saturated_oil_viscosity_beggs_robinson_ = saturated_oil_viscosity_beggs_r(rsb_m3m3, dead_oil_viscosity_)
            bubble_point_standing_ = bubble_point_valko_mccainsi(rsb_m3m3, gamma_gas, t, gamma_oil)
        ##Standing - debug, remove xxx
        ##p_bi = Bubblepoint_Standing(r_sb, gamma_g, t, gamma_o)
        ##Calculate bubble point correction factor
            if (p_rbcalc > 0):
        ##user specifie
                p_fact = bubble_point_standing_ / p_rbcalc
                p_offs = bubble_point_standing_ - p_rbcalc
            else:
        ##not specified, use from correlations
                p_fact = 1
                p_offs = 0

            if (oil_pressure_in_MPa_ > (bubble_point_standing_ / p_fact)):
        ##undersaturated oil
        ##apply correction to undersaturated oil
                oil_pressure_in_MPa_ = oil_pressure_in_MPa_ + p_offs
            else:
        ##apply correction to saturated oil
                oil_pressure_in_MPa_ = oil_pressure_in_MPa_ * p_fact

        ##Calculate oil formation volume factor correction factor
            if (bob_m3m3 > 0):
        ##user specified
                rho_o_sat = density_mccainsi(bubble_point_standing_, gamma_gas, t, gamma_oil, rsb_m3m3, bubble_point_standing_, compressibility_at_bubble_point_pressure_)
                b_o_sat = fvf_mccainsi(rsb_m3m3, gamma_gas, gamma_oil * rhoRef, rho_o_sat)
                b_fact = (bob_m3m3 - 1) / (b_o_sat - 1)
            else:
        ##not specified, use from correlations
                b_fact = 1

        ##Debug, uncomment xxx
            rs_m3m3 = gor_velardesi(oil_pressure_in_MPa_, bubble_point_standing_, gamma_gas, t, gamma_oil, rsb_m3m3)
        ##calculate compressibility at bubble point pressure
            compressibility_at_bubble_point_pressure_ = compressibility_oil_vb(rsb_m3m3, gamma_gas, t, gamma_oil, oil_pressure_in_MPa_)

            if (oil_pressure_in_MPa_ > bubble_point_standing_):
        ##undersaturated oil
        ##Debug, remove xxx
        ##r_si = r_sb
        ##apply correction to undersaturated oil
                rho_o_sat = density_mccainsi(bubble_point_standing_, gamma_gas, t, gamma_oil, rsb_m3m3, bubble_point_standing_, compressibility_at_bubble_point_pressure_)
                b_o_sat = fvf_mccainsi(rsb_m3m3, gamma_gas, gamma_oil * rhoRef, rho_o_sat)
                b_o_sat = b_fact * (b_o_sat - 1) + 1
        ##it is assumed that at pressure 1 atm bo = 1
                bo_m3m3 = b_o_sat * m.exp_(compressibility_at_bubble_point_pressure_ * (bubble_point_standing_ - oil_pressure_in_MPa_))
            else:
        ##Debug, remove xxx
        ##r_si = GOR_Standing(p_mpa, gamma_g, t, gamma_o)
        ##apply correction to saturated oil
                rho_o = density_mccainsi(oil_pressure_in_MPa_, gamma_gas, t, gamma_oil, rs_m3m3, bubble_point_standing_, compressibility_at_bubble_point_pressure_)
                bo_m3m3 = b_fact * (fvf_mccainsi(rs_m3m3, gamma_gas, gamma_oil * rhoRef, rho_o) - 1) + 1 ##it is assumed that at pressure 1 atm bo = 1

            if (rsb_m3m3 < 350):
        ##Calculate oil viscosity acoording to Standing
                mu_oil_cP = oil_viscosity_standing(rs_m3m3, dead_oil_viscosity_, oil_pressure_in_MPa_, bubble_point_standing_)
            else:
        ##Calculate according to BegsRobinson(saturated)and VasquezBegs(undersaturated)
                if (oil_pressure_in_MPa_ > bubble_point_standing_):
        ##undersaturated oil
                    mu_oil_cP = oil_viscosity_vasquez_beggs(saturated_oil_viscosity_beggs_robinson_, oil_pressure_in_MPa_, bubble_point_standing_)
                else:
        ##saturated oil
        ##Beggs Robinson
                    mu_oil_cP = saturated_oil_viscosity_beggs_r(rs_m3m3, dead_oil_viscosity_)

        if (PVT_CORRELATION == 2):
            ##Debug mode.Linear Rs and bo vs P, p_rb should be specified.
            ##gas properties
            ##ideal gas
            Zfactor = 1
            ##debug
            ##p_pr = p_mpa / p_pc
            ##T_pr = t / T_pc
            ##Zi = ZFactorDranchuk(T_pr, p_pr)
            bg_m3m3 = gas_fvf(t, oil_pressure_in_MPa_, Zfactor)
            mu_gas_cP = 0.0000000001
            ##Set to default.b_rb should be specified by user!
            p_fact = 1
            p_offs = 0
            bubble_point_standing_ = p_rbcalc
            if (oil_pressure_in_MPa_ > (bubble_point_standing_)):
                ##undersaturated oil
                rs_m3m3 = rsb_m3m3
            else:
                ##saturate
                rs_m3m3 = oil_pressure_in_MPa_ / p_rbcalc * rsb_m3m3
                ##r_si = GOR_Standing(p_mpa * p_fact, gamma_g, t, gamma_o)

            ##if b_o is not specified by the user then
            ##set b_o so, that oil density, recalculated with r_s would be equal to dead oil density
            if (bob_m3m3 < 0):
                if (gamma_oil == 0):
                    bob_m3m3 = None
                else:
                    bo_m3m3 = (1 + rs_m3m3 * (gamma_gas * rhoAir) / (gamma_oil * rhoRef))
            else:
                if (oil_pressure_in_MPa_ > bubble_point_standing_):
                    bo_m3m3 = bob_m3m3 ##undersaturated oil
                else:
                    bo_m3m3 = 1.0 + (bob_m3m3 - 1.0) * (oil_pressure_in_MPa_ - cPMpa) / (bubble_point_standing_ - cPMpa)
            mu_oil_cP = 1

        q_oil_rc_m3day, q_wat_rc_m3day, q_gas_rc_m3day = Q_fluid_calc(Qliq, water_сut_share, bo_m3m3, bw_m3m3, rp_m3m3, qgas_free_sm3day, rs_m3m3, bg_m3m3)
        if q_oil_rc_m3day == None: q_oil_rc_m3day = 0
        if q_wat_rc_m3day == None: q_wat_rc_m3day = 0
        if q_gas_rc_m3day == None: q_gas_rc_m3day = 0

        q_mix_rc_m3day = q_oil_rc_m3day + q_wat_rc_m3day + q_gas_rc_m3day
        qliq_rc_m3day = q_wat_rc_m3day + q_oil_rc_m3day
        fw_rc_fr = 0
        if qliq_rc_m3day > 0:
            fw_rc_fr = q_wat_rc_m3day / qliq_rc_m3day
        else:
            fw_rc_fr = water_сut_share
        mu_liq_cP = mu_oil_cP * (1 - fw_rc_fr) + mu_wat_cP * fw_rc_fr

        aPVTdic = calc_ST(pksep_atma, tIn_C, water_сut_share, aPVTdic)
        sigma_liq_Nm = aPVTdic['ST_liqgas_dyncm'] * 0.001
        aPVTdic.update({'sigma_liq_Nm': sigma_liq_Nm})

        rho_oil_rc_kgm3 = 1000 * (gamma_oil + rs_m3m3 * gamma_gas * const.const_rho_air / 1000) / bo_m3m3
        rho_wat_rc_kgm3 = 1000 * gamma_wat / bw_m3m3
        rho_liq_rc_kgm3 = (1 - water_сut_share) * rho_oil_rc_kgm3 + water_сut_share * rho_wat_rc_kgm3

        f_g = 0
        rho_gas_rc_kgm3 = 0
        if q_mix_rc_m3day > 0:
            f_g = q_gas_rc_m3day / q_mix_rc_m3day
        if bg_m3m3 is not None and bg_m3m3 > 0:
            rho_gas_rc_kgm3 = gamma_gas * const.const_rho_air / bg_m3m3

        rho_mix_rc_kgm3 = rho_liq_rc_kgm3 * (1 - f_g) + rho_gas_rc_kgm3 * f_g
    except:
#        log.logger.exception(f'Module: "{__name__}"       Function: "{calc_pvt_vr.__name__}"')
#        log.logger.error(f'Module: "{__name__}"       Function: "{calc_pvt_vr.__name__}"       Parameters: pksep_atma = {pksep_atma}, tIn_C = {tIn_C}, Qliq = {Qliq}, water_сut_share = {water_сut_share}')
        return aPVTdic
    ##Assign output variables
        # расчетное значение давления насыщения по корреляции
    pbcalc_atma = bubble_point_standing_ / cPMpa / p_fact
    aPVTdic.update({'pbcalc_atma': pbcalc_atma})
        # расчетное значение газосодержания в нефти при текущих условиях
    aPVTdic.update({'rs_m3m3': rs_m3m3})
        # объемный коэффициент нефти при рабочих условиях
    aPVTdic.update({'bo_m3m3': bo_m3m3})
        # расчетное значение коэффициента сверхсжимаемости газа
    aPVTdic.update({'Zfactor': Zfactor})
        # объемный коэффициент газа при известном коэффициенте сжимаемости газа
    aPVTdic.update({'bg_m3m3': bg_m3m3})
        # вязкость нефти при рабочих условиях
    aPVTdic.update({'mu_oil_cP': mu_oil_cP})
        # вязкость воды при рабочих условиях
    aPVTdic.update({'mu_wat_cP': mu_wat_cP})
        # вязкость газа при рабочих условиях
    aPVTdic.update({'mu_gas_cP': mu_gas_cP})
        # вязкость жидкости при рабочих условиях
    aPVTdic.update({'mu_liq_cP': mu_gas_cP})

        # объемный коэффициент воды
    aPVTdic.update({'bw_m3m3': bw_m3m3})

    aPVTdic.update({'q_oil_rc_m3day': q_oil_rc_m3day})
    aPVTdic.update({'q_wat_rc_m3day': q_wat_rc_m3day})
    aPVTdic.update({'q_gas_rc_m3day': q_gas_rc_m3day})
    aPVTdic.update({'qliq_rc_m3day': qliq_rc_m3day})

    aPVTdic.update({'q_mix_rc_m3day': q_mix_rc_m3day})

    aPVTdic.update({'rho_oil_rc_kgm3': rho_oil_rc_kgm3})
    aPVTdic.update({'rho_wat_rc_kgm3': rho_wat_rc_kgm3})
    aPVTdic.update({'rho_liq_rc_kgm3': rho_liq_rc_kgm3})
    aPVTdic.update({'rho_gas_rc_kgm3': rho_gas_rc_kgm3})
    aPVTdic.update({'rho_mix_rc_kgm3': rho_mix_rc_kgm3})

    return aPVTdic