from abc import ABC, abstractmethod
import pandas as pd
import math_ext as m

class calc_pvt(ABC):
    def __init__(self, PVTCorr = 0, gamma_gas=0.6, gamma_oil=0.86, gamma_wat=1, ksep_fr = 0.7, rsb_m3m3=100, rp_m3m3=100, pb_atma=-1,
                 tres_C=90, bob_m3m3=-1, qgas_free_sm3day=0, muob_cP=-1, bwSC_m3m3=0, salinity_ppm=0):

        self.PVTCorr = PVTCorr
        self.gamma_gas = gamma_gas
        self.gamma_oil = gamma_oil
        self.gamma_wat = gamma_wat
        self.ksep_fr = ksep_fr
        self.rsb_m3m3 = rsb_m3m3
        self.rp_m3m3 = rp_m3m3
        self.pb_atma = pb_atma
        self.tres_C = tres_C
        self.bob_m3m3 = bob_m3m3
        self.qgas_free_sm3day = qgas_free_sm3day
        self.muob_cP = muob_cP
        self.bwSC_m3m3 = bwSC_m3m3
        self.salinity_ppm = salinity_ppm


#    __gamma_gas = None
    ''' плотность газа удельная '''

    @property
    def gamma_gas(self):
        return self.__gamma_gas

    @gamma_gas.setter
    def gamma_gas(self, gamma_gas):
        self.__gamma_gas = gamma_gas

#    __gamma_oil = None
    ''' плотность нефти удельная '''

    @property
    def gamma_oil(self):
        return self.__gamma_oil

    @gamma_oil.setter
    def gamma_oil(self, gamma_oil):
        self.__gamma_oil = gamma_oil

#    __gamma_wat = None
    ''' плотность воды удельная '''

    @property
    def gamma_wat(self):
        return self.__gamma_wat

    @gamma_wat.setter
    def gamma_wat(self, gamma_wat):
        self.__gamma_wat = gamma_wat
#        if self.bwSC_m3m3 is None:
#            self.bwSC_m3m3 = gamma_wat / 1

#    __rsb_m3m3 = None
    ''' газосодержание при давлении насыщения '''

    @property
    def ksep_fr(self):
        return self.__ksep_fr

    @ksep_fr.setter
    def ksep_fr(self, ksep_fr):
        self.__ksep_fr = ksep_fr

    @property
    def rsb_m3m3(self):
        return self.__rsb_m3m3

    @rsb_m3m3.setter
    def rsb_m3m3(self, rsb_m3m3):
        self.__rsb_m3m3 = rsb_m3m3

#    __rp_m3m3 = None
    ''' Газовый фактор добычной в стандартных условиях '''

    @property
    def rp_m3m3(self):
        return self.__rp_m3m3

    @rp_m3m3.setter
    def rp_m3m3(self, rp_m3m3):
        self.__rp_m3m3 = rp_m3m3

#    __pb_atma = None
    ''' давление насыщения  (калибровочное значение) '''

    @property
    def pb_atma(self):
        return self.__pb_atma

    @pb_atma.setter
    def pb_atma(self, pb_atma):
        self.__pb_atma = pb_atma

#    __tres_C = None
    ''' температура пласта, C '''

    @property
    def tres_C(self):
        return self.__tres_C

    @tres_C.setter
    def tres_C(self, tres_C):
        self.__tres_C = tres_C

#    __bob_m3m3 = None
    ''' объемный коэффициент при давлении насыщения '''

    @property
    def bob_m3m3(self):
        return self.__bob_m3m3

    @bob_m3m3.setter
    def bob_m3m3(self, bob_m3m3):
        self.__bob_m3m3 = bob_m3m3

#    __muob_cP = None
    ''' вязкость нефти при давлении насыщения (калибровочное значение) '''

    @property
    def muob_cP(self):
        return self.__muob_cP

    @muob_cP.setter
    def muob_cP(self, muob_cP):
        self.__muob_cP = muob_cP

#    __bwSC_m3m3 = None

#    @property
#    def bwSC_m3m3(self):
#        return self.__bwSC_m3m3

#    @bwSC_m3m3.setter
#    def bwSC_m3m3(self, bwSC_m3m3):
#        if self.__bwSC_m3m3 is None:
#            self.__bwSC_m3m3 = bwSC_m3m3

    ''' соленость воды '''
#    __salinity_ppm = None

    @property
    def salinity_ppm(self):
        return self.__salinity_ppm

    @salinity_ppm.setter
    def salinity_ppm(self, salinity_ppm):
        self.__salinity_ppm = salinity_ppm

    def get_summary_df_pvt(self):
        pvt_index = ['Источники', 'Единиц.изм', 'Значения']

        summary_df_pvt = pd.DataFrame(index=pvt_index, columns=['Плотность газа попутного относительная',
                                                                'Удельный вес сепарированной нефти',
                                                                'Удельный вес попутной воды',
                                                                'Давление насыщения нефти газом',
                                                                'Температура пласта',
                                                                'Объемный коэффициент нефти',
                                                                'Динамическая вязкость сепарированной нефти',
                                                                'Газосодержание, м3/м3, станд.сепар.',
                                                                'Газовый фактоp общий'
                                                                ])
        summary_df_pvt['Плотность газа попутного относительная'] = ['sppl_sk.pg_1', '', self.gamma_gas]
        summary_df_pvt['Удельный вес сепарированной нефти'] = ['sppl_sk.pn_1', 'т/м3', self.gamma_oil]
        summary_df_pvt['Удельный вес попутной воды'] = ['sppl_sk.pv_1', 'т/м3', self.gamma_wat]
        summary_df_pvt['Давление насыщения нефти газом'] = ['sppl_sk.dn_1', 'атм', self.pb_atma]
        summary_df_pvt['Температура пласта'] = ['sppl_sk.tl_1', 'град С', self.tres_C]
        summary_df_pvt['Объемный коэффициент нефти'] = ['sppl_sk.ok_1', '', self.bob_m3m3]
        summary_df_pvt['Динамическая вязкость сепарированной нефти'] = ['sppl_sk.vn_1', 'спз', self.muob_cP]
        summary_df_pvt['Газосодержание, м3/м3, станд.сепар.'] = ['Константа', 'м3/м3', self.rsb_m3m3]
        summary_df_pvt['Газовый фактоp общий'] = ['sppl_sk.fo_1', 'нм3/т', self.rp_m3m3]
        return summary_df_pvt

    def properties_to_str(self):
        str = f"gamma_gas = {self.gamma_gas}, gamma_oil = {self.gamma_oil}, gamma_wat = {self.gamma_wat}, " \
              f"rsb_m3m3 = {self.rsb_m3m3}, rp_m3m3 = {self.rp_m3m3}, pb_atma = {self.pb_atma}, " \
              f"tres_C = {self.tres_C}, bob_m3m3 = {self.bob_m3m3}, muob_cP = {self.muob_cP}, " \
              f"bwSC_m3m3 = {self.bwSC_m3m3}, salinity_ppm = {self.salinity_ppm}"
        return str

    def gas_fvf(self, temperature, pump_presure, z):
        # Расчет объемного коэффициента газа при известном коэффиенте сжимаемости газа
        if temperature != temperature or (pump_presure == 0) or pump_presure != pump_presure or z != z: return None
        return 0.00034722 * temperature * z / pump_presure

    def pseudo_pressure(self, gg):
        if ((4.9 - 0.4 * gg) == 0): return None
        return 4.9 - 0.4 * gg

    def pseudo_temperature(self, gg):
        return 95 + 171 * gg

    def pseudo_temperature_standing(self, gg):
        return 93.3 + 180 * gg - 6.94 * m.pow_(gg, 2)

    def pseudo_pressure_standing(self, gg):
        return 4.6 + 0.1 * gg - 0.258 * m.pow_(gg, 2)

    def dead_oil_viscosity_beggs_robin(self, temperature_k, gamma_oil):
        if (gamma_oil is None or gamma_oil == 0): return None
        x = m.pow_((1.8 * temperature_k - 460), -1.163) * m.exp_(13.108 - 6.591 / gamma_oil)
        return m.pow_(10, x) - 1

    def saturated_oil_viscosity_beggs_r(self, gor_pb_m3m3, dead_oil_viscosity):
        if gor_pb_m3m3 != gor_pb_m3m3 or dead_oil_viscosity != dead_oil_viscosity:
            return None
        a = 10.715 * m.pow_(5.615 * gor_pb_m3m3 + 100, -0.515)
        b = 5.44 * m.pow_(5.615 * gor_pb_m3m3 + 150, -0.338)
        return a * m.pow_(dead_oil_viscosity, b)

    def water_fvf(self, pressure_mpa, temperature_k):
        # Расчет объемного коэффициента воды
        if pressure_mpa != pressure_mpa or temperature_k != temperature_k:
            return None
        f = 1.8 * float(temperature_k) - 460
        psi = float(pressure_mpa) * 145.04
        dvwp = -1.95301 * m.pow_(10, -9) * psi * f - 1.72834 * m.pow_(10, -13) * m.pow_(psi, 2) * f - 3.58922 * m.pow_(
            10, -7) * psi - 2.25341 * m.pow_(10, -10) * m.pow_(psi, 2)
        dvwt = -1.0001 * m.pow_(10, -2) + 1.33391 * m.pow_(10, -4) * f + 5.50654 * m.pow_(10, -7) * m.pow_(f, 2)
        return (1 + dvwp) * (1 + dvwt)

    def water_viscosity(self, pressure_mpa, temperature_k, salinity_mg_liter):
        # Расчет вязкости воды
        if pressure_mpa != pressure_mpa or temperature_k != temperature_k:
            return None
        wptds = salinity_mg_liter / 10000
        a = 109.574 - 8.40564 * wptds + 0.313314 * m.pow_(wptds, 2) + 0.00872213 * m.pow_(wptds, 3)
        b = -1.12166 + 0.0263951 * wptds - 0.000679461 * m.pow_(wptds, 2) - 5.47119 * m.pow_(10, -5) * m.pow_(wptds,
                                                                                                              3) + 1.55586 * m.pow_(
            10, -6) * m.pow_(wptds, 4)
        visc = a * m.pow_(1.8 * temperature_k - 460, b)
        psi = pressure_mpa * 145.04
        return visc * (0.9994 + 4.0295 * m.pow_(10, -5) * psi + 3.1062 * m.pow_(10, -9) * m.pow_(psi, 2))

    def g_visc(self, t, p, z, gg):
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

    def bubble_point_standing(self, rsb_m3m3, gamma_gas, temperature_k, gamma_oil):
        min_rsb = 1.8
        _bubblepoint_standing = 0
        yg = 0
        rsb_old = rsb_m3m3
        rsb_m3m3calc = rsb_m3m3

        if (rsb_m3m3calc < min_rsb):
            rsb_m3m3calc = min_rsb
        if (gamma_oil == 0): return None

        yg = 1.225 + 0.001648 * temperature_k - 1.769 / gamma_oil
        _bubblepoint_standing = 0.5197 * m.pow_(rsb_m3m3calc / gamma_gas, 0.83) * m.pow_(10, yg)

        if (rsb_old < min_rsb):
            _bubblepoint_standing = (_bubblepoint_standing - 0.1013) * rsb_old / min_rsb + 0.1013
        return _bubblepoint_standing

    def fvf_saturated_oil_standing(self, rs_m3m3, gamma_gas, temperature_k, gamma_oil):
        if (gamma_oil == 0): return None
        f = 5.615 * rs_m3m3 * m.pow_(gamma_gas / gamma_oil, 0.5) + 2.25 * temperature_k - 575
        return 0.972 + 0.000147 * m.pow_(f, 1.175)

    def compressibility_oil_vb(self, rs_m3m3, gamma_gas, temperature_k, gamma_oil, pressure_mpa):
        if (gamma_oil == 0 or pressure_mpa == 0): return None
        return (28.1 * rs_m3m3 + 30.6 * temperature_k - 1180 * gamma_gas + 1784 / gamma_oil - 10910) / (
                    100000 * pressure_mpa)

    def oil_viscosity_vasquez_beggs(self, saturated_oil_viscosity, pressure_mpa, bp_pressure_mpa):
        c1 = 957
        c2 = 1.187
        c3 = -11.513
        c4 = -0.01302
        if (pressure_mpa is None or bp_pressure_mpa == 0): return None
        pow_gradle = c1 * m.pow_(pressure_mpa, c2) * m.exp_(c3 + c4 * pressure_mpa)
        return saturated_oil_viscosity * m.pow_(pressure_mpa / bp_pressure_mpa, pow_gradle)

    def gor_standing(sefl, pressure_mpa, gamma_gas, temperature_k, gamma_oil):
        if (gamma_oil == 0): return None
        yg = 1.225 + 0.001648 * temperature_k - 1.769 / gamma_oil
        return gamma_gas * m.pow_(1.92 * pressure_mpa / m.pow_(10, yg), 1.204)

    def zfactor(self, tpr, ppr):
        if (ppr is None or tpr is None or tpr == 0.86): return None
        a = 1.39 * m.pow_((tpr - 0.92), 0.5) - 0.36 * tpr - 0.101
        b = ppr * (0.62 - 0.23 * tpr) + m.pow_(ppr, 2) * (0.006 / (tpr - 0.86) - 0.037) + 0.32 * m.pow_(ppr,
                                                                                                        6) / m.exp_(
            20.723 * (tpr - 1))
        if (b is None): return None
        c = 0.132 - 0.32 * m.log_(tpr) / m.log_(10)
        d = m.exp_(0.715 - 1.128 * tpr + 0.42 * m.pow_(tpr, 2))
        if (a is None or m.exp_(-b) is None or c is None or m.pow_(ppr, d) is None): return None
        result = a + (1 - a) * m.exp_(-b) + c * m.pow_(ppr, d)
        return result

    def dead_oil_viscosity_standing(self, temperature_k, gamma_oil):
        if (gamma_oil is None or temperature_k is None or gamma_oil == 0 or (141.5 / gamma_oil - 131.5) == 0 or (
                1.8 * temperature_k - 260) == 0):
            return None
        result = (0.32 + 1.8 * m.pow_(10, 7) / m.pow_(141.5 / gamma_oil - 131.5, 4.53)) * m.pow_(
            360 / (1.8 * temperature_k - 260), m.pow_(10, (0.43 + 8.33 / (141.5 / gamma_oil - 131.5))))
        return result

    def fvf_mccainsi(self, rs_m3m3, gamma_gas, sto_density_kg_m3, reservoir_oil_density_kg_m3):
        if (reservoir_oil_density_kg_m3 == 0): return None
        return (sto_density_kg_m3 + 1.22117 * rs_m3m3 * gamma_gas) / reservoir_oil_density_kg_m3

    def oil_viscosity_standing(self, rs_m3m3, dead_oil_viscosity, pressure_mpa, bubblepoint_pressure_mpa):
        a = 5.6148 * rs_m3m3 * (0.1235 * m.pow_(10, -5) * rs_m3m3 - 0.00074)
        b = 0.68 / m.pow_(10, 0.000484 * rs_m3m3) + 0.25 / m.pow_(10, 0.006176 * rs_m3m3) + 0.062 / m.pow_(10,
                                                                                                           0.021 * rs_m3m3)
        _oil_viscosity_standing = m.pow_(10, a) * m.pow_(dead_oil_viscosity, b)
        if (bubblepoint_pressure_mpa < pressure_mpa):
            _oil_viscosity_standing = _oil_viscosity_standing + 0.14504 * (pressure_mpa - bubblepoint_pressure_mpa) * (
                        0.024 * m.pow_(_oil_viscosity_standing, 1.6) + 0.038 * m.pow_(_oil_viscosity_standing, 0.56))
        return _oil_viscosity_standing

    def bubble_point_valko_mccainsi(self, rsb_m3m3, gamma_gas, temperature_k, gamma_oil):
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
        if (
                rsb_m3m3 is None or gamma_gas is None or temperature_k is None or gamma_oil is None or gamma_oil == 0): return None
        rsb_m3m3calc = rsb_m3m3
        rsb_old = rsb_m3m3calc
        rsb_m3m3calc = rsb_m3m3

        if (rsb_m3m3calc < min_rsb): rsb_m3m3calc = min_rsb
        if (rsb_m3m3calc > max_rsb): rsb_m3m3calc = max_rsb;

        api = 141.5 / gamma_oil - 131.5
        z1 = -4.814074834 + 0.7480913 * m.log_(rsb_m3m3calc) + 0.1743556 * m.pow_(m.log_(rsb_m3m3calc),
                                                                                  2) - 0.0206 * m.pow_(
            m.log_(rsb_m3m3calc), 3)
        z2 = 1.27 - 0.0449 * api + 4.36 * m.pow_(10, -4) * m.pow_(api, 2) - 4.76 * m.pow_(10, -6) * m.pow_(api, 3)
        z3 = 4.51 - 10.84 * gamma_gas + 8.39 * m.pow_(gamma_gas, 2) - 2.34 * m.pow_(gamma_gas, 3)
        z4 = -7.2254661 + 0.043155 * temperature_k - 8.5548 * m.pow_(10, -5) * m.pow_(temperature_k,
                                                                                      2) + 6.00696 * m.pow_(10,
                                                                                                            -8) * m.pow_(
            temperature_k, 3)
        z = z1 + z2 + z3 + z4
        lnpb = 2.498006 + 0.713 * z + 0.0075 * z * z
        _bubblepoint_valko_mccainsi = m.pow_(2.718282, lnpb)

        if (rsb_old < min_rsb): _bubblepoint_valko_mccainsi = (_bubblepoint_valko_mccainsi - 0.1013) * rsb_old / min_rsb + 0.1013

        if (rsb_old > max_rsb): _bubblepoint_valko_mccainsi = (_bubblepoint_valko_mccainsi - 0.1013) * rsb_old / max_rsb + 0.1013

        return _bubblepoint_valko_mccainsi

    def gor_velardesi(self, pressure_mpa, bubblepoint_pressure_mpa, gamma_gas, temperature_k, gamma_oil, rsb_m3_m3):
        maxrs = 800
        api = gor_velardesi = pr = a_0 = a_1 = a_2 = a_3 = a_4 = b_0 = b_1 = b_2 = b_3 = b_4 = 0
        c_0 = c_1 = c_2 = c_3 = c_4 = a1 = a2 = a3 = rsr = 0

        if (gamma_oil == 0): return None
        api = 141.5 / gamma_oil - 131.5
        if (bubblepoint_pressure_mpa > self.bubble_point_valko_mccainsi(maxrs, gamma_gas, temperature_k, gamma_oil)):
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
            gor_velardesi = rsb_m3_m3

        if (pr < 1):
            a_0 = 1.8653 * m.pow_(10, -4)
            a_1 = 1.672608
            a_2 = 0.92987
            a_3 = 0.247235
            a_4 = 1.056052
            a1 = a_0 * m.pow_(gamma_gas, a_1) * m.pow_(api, a_2) * m.pow_(1.8 * temperature_k - 460, a_3) * m.pow_(
                bubblepoint_pressure_mpa, a_4)
            b_0 = 0.1004
            b_1 = -1.00475
            b_2 = 0.337711
            b_3 = 0.132795
            b_4 = 0.302065
            a2 = b_0 * m.pow_(gamma_gas, b_1) * m.pow_(api, b_2) * m.pow_(1.8 * temperature_k - 460, b_3) * m.pow_(
                bubblepoint_pressure_mpa, b_4)
            c_0 = 0.9167
            c_1 = -1.48548
            c_2 = -0.164741
            c_3 = -0.09133
            c_4 = 0.047094
            a3 = c_0 * m.pow_(gamma_gas, c_1) * m.pow_(api, c_2) * m.pow_(1.8 * temperature_k - 460, c_3) * m.pow_(
                bubblepoint_pressure_mpa, c_4)
            rsr = a1 * m.pow_(pr, a2) + (1 - a1) * m.pow_(pr, a3)
            gor_velardesi = rsr * rsb_m3_m3
        return gor_velardesi

    def density_mccainsi(self, pressure_mpa, gamma_gas, temperature_k, gamma_oil, rs_m3_m3, bp_pressure_mpa, compressibility):
        maxiter = 1000
        rs_m3_m3calc = ropo = pm = pmmo = epsilon = a0 = a1 = a2 = a3 = a4 = a5 = 0
        _density_mccainsi = dpt = roa = dpp = pbs = bp_pressure_mpacalc = 0

        rs_m3_m3calc = rs_m3_m3
        bp_pressure_mpacalc = bp_pressure_mpa

        if (rs_m3_m3calc > 800):
            rs_m3_m3calc = 800
            bp_pressure_mpacalc = self.bubble_point_valko_mccainsi(rs_m3_m3calc, gamma_gas, temperature_k, gamma_oil)

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
            roa = a0 + a1 * gamma_gas + a2 * gamma_gas * ropo + a3 * gamma_gas * m.pow_(ropo,
                                                                                        2) + a4 * ropo + a5 * m.pow_(
                ropo, 2)
            ropo = (rs_m3_m3calc * gamma_gas + 818.81 * gamma_oil) / (0.81881 + rs_m3_m3calc * gamma_gas / roa)
            pm = ropo
            counter = counter + 1

        if (pressure_mpa <= bp_pressure_mpacalc):
            dpp = (0.167 + 16.181 * m.pow_(10, (-0.00265 * pm))) * (2.32328 * pressure_mpa) - 0.16 * (
                        0.299 + 263 * m.pow_(10, (-0.00376 * pm))) * m.pow_(0.14503774 * pressure_mpa, 2)
            pbs = pm + dpp
            dpt = (0.04837 + 337.094 * m.pow_(pbs, -0.951)) * m.pow_(1.8 * temperature_k - 520, 0.938) - (
                        0.346 - 0.3732 * m.pow_(10, -0.001 * pbs)) * m.pow_(1.8 * temperature_k - 520, 0.475)
            pm = pbs - dpt
            _density_mccainsi = pm
        else:
            dpp = (0.167 + 16.181 * m.pow_(10, -0.00265 * pm)) * (2.32328 * bp_pressure_mpacalc) - 0.16 * (
                        0.299 + 263 * m.pow_(10, -0.00376 * pm)) * m.pow_(0.14503774 * bp_pressure_mpacalc, 2)
            pbs = pm + dpp
            dpt = (0.04837 + 337.094 * m.pow_(pbs, -0.951)) * m.pow_(1.8 * temperature_k - 520, 0.938) - (
                        0.346 - 0.3732 * m.pow_(10, -0.001 * pbs)) * m.pow_(1.8 * temperature_k - 520, 0.475)
            pm = pbs - dpt
            _density_mccainsi = pm * m.exp_(compressibility * (pressure_mpa - bp_pressure_mpacalc))
        return _density_mccainsi

    def unf_calc_Sal_BwSC_ppm(self, BwSC):
        # функция для оценки солености воды по объемному коэффициенту (получена как обратная к unf_calc_BwSC_d)
        salinity = m.sqrt_(624.711071129603 * BwSC / 0.0160185 - 20192.9595437054) - 137.000074965329
        return salinity * 10000

    def Q_fluid_calc(self, qliq_sm3day, waterСut, RoilVol, RwatVol, rp_m3m3, qgas_free_sm3day, RgasVolOil, RgasVol):
        # Расчет дебитов нефти (qOil), воды (qWat), газа (qGas)
        # qliq_sm3day      - дебит жидкости
        # waterСut         - Обводненность, доля не процент
        # RoilVol          - объемный коэффициент нефти при рабочих условиях
        # RwatVol          - объемный коэффициент воды
        # rp_m3m3          - газовый фактор добычной (приведенный к стандартным условиям)
        # qgas_free_sm3day - свободный газ в потоке
        # RgasVolOil       - расчетное значение газосодержания в нефти при текущих условиях
        # RgasVol          - объемный коэффициент газа при известном коэффициенте сжимаемости газа

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
            q_gas_rc_m3day = (qliq_sm3day * (1 - waterСut) * rp_m3m3 + qgas_free_sm3day - RgasVolOil * qliq_sm3day * (
                        1 - waterСut)) * RgasVol
        return (q_oil_rc_m3day, q_wat_rc_m3day, q_gas_rc_m3day)

    def getoil_API(self, gamma_oil):
        return 141.5 / gamma_oil - 131.5

    def calc_ST(self, p_atma, t_C, water_сut_share, gamma_oil):
        # water_сut_share     - Обводненность, доля не процент
        ST68 = None
        ST100 = None
        STw74 = None
        STw280 = None
        Tst = None
        Tstw = None
        ST_oilgas_dyncm = 0
        ST_watgas_dyncm = 0
        ST_liqgas_dyncm = 0
        t_F = None
        P_psia = None
        P_MPa = None
        # calculate surface tension according Baker Sverdloff correlation
        # VB2PY (UntranslatedCode) On Error GoTo err1
        # Расчет коэффициента поверхностного натяжения газ-нефть
        t_F = t_C * 1.8 + 32
        P_psia = p_atma / 0.068046
        P_MPa = p_atma / 10
        oil_API = self.getoil_API(gamma_oil)
        ST68 = 39 - 0.2571 * oil_API
        ST100 = 37.5 - 0.2571 * oil_API
        if t_F < 68:
            ST_oilgas_dyncm = ST68
        else:
            Tst = t_F
            if t_F > 100:
                Tst = 100
            ST_oilgas_dyncm = (68 - (((Tst - 68) * (ST68 - ST100)) / 32)) * (1 - (0.024 * (P_psia) ** 0.45))
            if ST_oilgas_dyncm < 0:
                ST_oilgas_dyncm = ST68
                # todo rnt20190312 надо будет исправить когда то
        # Расчет коэффициента поверхностного натяжения газ-вода  (два способа)
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
        ST_watgas_dyncm = 10 ** (-(1.19 + 0.01 * P_MPa)) * 1000
        # Расчет коэффициента поверхностного натяжения газ-жидкость
        ST_liqgas_dyncm = (ST_watgas_dyncm * water_сut_share) + ST_oilgas_dyncm * (1 - water_сut_share)
        return ST_oilgas_dyncm, ST_watgas_dyncm, ST_liqgas_dyncm

    _q_liq = 0
    _tIn_C = 0
    _water_сut_share = 0

    @abstractmethod
    def calc(self, pksep_atma, q_liq, tIn_C, water_сut_share, z_factor_func, pseudo_crit_func):
        self._pksep_atma = pksep_atma
        self._q_liq = q_liq
        self._tIn_C = tIn_C
        self._water_сut_share = water_сut_share
        pass

















