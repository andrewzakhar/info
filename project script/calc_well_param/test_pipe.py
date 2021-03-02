import unittest

import cpipe
import ccalc_pvt
from CPVT import standing

class CPipeTestCase(unittest.TestCase):
    def test_pipe_calc(self):
        # Вынгаяхинское месторождение
        gamma_oil_ = 0.82
        gamma_wat_ = 1
        gamma_gas_ = 0.796
        Rsb_ = 15.6
        Rp_ = 124
        Pb_ = 139
        Tres_ = 89
        Bob_ = 1.278
        mu_ = 7.76 # Вязкость нефти, из PVT

        PVTdic = {}
        PVTdic['PVTcorr'] = 0
        PVTdic['ksep_fr'] = float(0.7)
        PVTdic['qgas_free_sm3day'] = 0

        PVTdic['gamma_gas'] = float(gamma_gas_)
        PVTdic['gamma_oil'] = float(gamma_oil_)
        PVTdic['gamma_wat'] = float(gamma_wat_)
        PVTdic['rsb_m3m3'] = float(Rsb_)
        PVTdic['rp_m3m3'] = float(7.071955128205128)
        PVTdic['pb_atma'] = float(Pb_)
        PVTdic['tres_C'] = float(Tres_)
        PVTdic['bob_m3m3'] = float(Bob_)
        PVTdic['muob_cP'] = float(mu_)
        PVTdic['bwSC_m3m3'] = PVTdic['gamma_wat'] / 1
        PVTdic['salinity_ppm'] = ccalc_pvt.unf_calc_Sal_BwSC_ppm(PVTdic['bwSC_m3m3'])
        PVTdic['mu_liq_cP'] = 0.837091367843072
        PVTdic['mu_gas_cP'] = 0.0124383986806129
        PVTdic['sigma_liq_Nm'] = 0.0640952157612172
        PVTdic['rho_liq_rc_kgm3'] = 949.132237730242
        PVTdic['rho_gas_rc_kgm3'] = 1.20723236536855
        PVTdic['qliq_rc_m3day'] = 65.2660317740851
        PVTdic['q_gas_rc_m3day'] = 8.48031982336487

        Qliq_ = 62.4
        Fw_ = 94.4
        Length_ = 10.0
        Pcalc_ = 1.536
        Calc_along_flow_ = 0
        Theta_deg = 89.95
        Dtub_ = 79
        Hydr_corr_ = 0
        Tcalc_ = 89
        Tother_ = 89        
        pressure = cpipe.pipe_atma(Qliq_, Fw_, Length_, Pcalc_, Calc_along_flow_, PVTdic, Theta_deg, Dtub_, Hydr_corr_, Tcalc_, Tother_)
        self.assertAlmostEqual(2.4288745371094422, pressure[0], places=1)

# 2.404875317243335 != 2.4288745371094422
    #
    # Expected :2.4288745371094422
    # Actual   :2.404875317243335

    def test_pipe_calc_clpvt(self):
        # Вынгаяхинское месторождение
        gamma_oil = 0.82
        gamma_wat = 1
        gamma_gas = 0.796
        rsb_m3m3 = 15.6
        rp_m3m3 = float(7.071955128205128)
        pb_atma = 139
        tres_C = 89
        t_ksep_C = 90
        bob_m3m3 = 1.278
        muob_cP = 7.76  # Вязкость нефти, из PVT
        bwSC_m3m3 = gamma_wat / 1
        PVTCorr = 0
        ksep_fr = 0.7
        qgas_free_sm3day = 0
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)
        class_pvt = standing.calc_standing()
        class_pvt.PVTCorr = PVTCorr
        class_pvt.gamma_oil = gamma_oil
        class_pvt.gamma_wat = gamma_wat
        class_pvt.gamma_gas = gamma_gas
        class_pvt.rsb_m3m3 = rsb_m3m3
        class_pvt.rp_m3m3 = rp_m3m3
        class_pvt.pb_atma = pb_atma
        class_pvt.tres_C = tres_C
        class_pvt.bob_m3m3 = bob_m3m3
        class_pvt.t_ksep_C = t_ksep_C
        class_pvt.muob_cP = muob_cP  # Вязкость нефти, из PVT
        class_pvt.qgas_free_sm3day = qgas_free_sm3day
        class_pvt.ksep_fr = ksep_fr
        class_pvt.bwSC_m3m3 = bwSC_m3m3
        class_pvt.salinity_ppm = salinity_ppm

        Qliq_ = 62.4
        Fw_ = 94.4
        Length_ = 10.0
        Pcalc_ = 1.536
        Calc_along_flow_ = 0
        Theta_deg = 89.95
        Dtub_ = 79
        Hydr_corr_ = 0
        Tcalc_ = 89
        Tother_ = 89
        pressure = cpipe.pipe_atma_pvtcls(Qliq_, Fw_, Length_, Pcalc_, Calc_along_flow_, class_pvt, Theta_deg, Dtub_,
                                          Hydr_corr_, Tcalc_, Tother_)
        self.assertAlmostEqual(2.4288745371094422, pressure[0], places=1)

# 2.405084766272263 != 2.4288745371094422
#
# Expected :2.4288745371094422
# Actual   :2.405084766272263

    def test_pipe_2calc_clpvt(self):
        # Вынгаяхинское месторождение
        gamma_oil = 0.777
        gamma_wat = 1.054
        gamma_gas = 0.749
        rsb_m3m3 = float(2.61984807966408)
        rp_m3m3 = float(2.61984807966408)
        pb_atma = float(106.754372911581)
        tres_C = 101.5
        bob_m3m3 = 1.51159396950742
        muob_cP = 5.58  # Вязкость нефти, из PVT
        bwSC_m3m3 = gamma_wat / 1
        PVTCorr = 0
        ksep_fr = 0.7
        qgas_free_sm3day = 0
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)
        class_pvt = standing.calc_standing()
        class_pvt.PVTCorr = PVTCorr
        class_pvt.gamma_oil = gamma_oil
        class_pvt.gamma_wat = gamma_wat
        class_pvt.gamma_gas = gamma_gas
        class_pvt.rsb_m3m3 = rsb_m3m3
        class_pvt.rp_m3m3 = rp_m3m3
        class_pvt.pb_atma = pb_atma
        class_pvt.tres_C = tres_C
        class_pvt.bob_m3m3 = bob_m3m3
#        class_pvt.t_ksep_C = t_ksep_C
        class_pvt.muob_cP = muob_cP  # Вязкость нефти, из PVT
        class_pvt.qgas_free_sm3day = qgas_free_sm3day
        class_pvt.ksep_fr = ksep_fr
        class_pvt.bwSC_m3m3 = bwSC_m3m3
        class_pvt.salinity_ppm = salinity_ppm

        Qliq_ = 62.4
        Fw_ = 94.4
        Length_ = 10.0
        Pcalc_ = 1.536
        Calc_along_flow_ = 0
        Theta_deg = 89.95
        Dtub_ = 79
        Hydr_corr_ = 0
        Tcalc_ = 101.5
        Tother_ = 101.5
        pressure = cpipe.pipe_atma_pvtcls(Qliq_, Fw_, Length_, Pcalc_, Calc_along_flow_, class_pvt, Theta_deg, Dtub_,
                                          Hydr_corr_, Tcalc_, Tother_)
        self.assertEqual(2.46295286730971, pressure[0])

        # 2.472686974159582 != 2.46295286730971
        # 2.4629528673097094 != 2.46295286730971

