import unittest
import calc_func
import ccalc_pvt
import standing
import mc_cain
import straight_line

class pvt_class_test(unittest.TestCase):
    def test_calc_standing(self):
        class_pvt = standing.calc_standing()
        gamma_oil = 0.777
        gamma_wat = 1.054
        gamma_gas = 0.749
        rsb_m3m3 = 2.20773800714468
        rp_m3m3 = 2.20773800714468
        pb_atma = 106.648752781247
        tres_C = 101.5
        bob_m3m3 = 1.52086750956118
        muob_cP = 5.58  # Вязкость нефти, из PVT
        qgas_free_sm3day = 0
        tksep_C = 90
        bwSC_m3m3 = gamma_wat / 1
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)

        Pintake_ = 34.5
        Qliq_ = 61.2
        Fw_ = 96.7
        ksep_fr = 0.7
        Tintake_ = 90
        z_class = calc_func.z_factor_2015_kareem()
        pseudo_class = calc_func.pseudo_standing()

        class_pvt.gamma_gas = gamma_gas
        class_pvt.gamma_oil = gamma_oil
        class_pvt.gamma_wat = gamma_wat
        class_pvt.ksep_fr = ksep_fr
        class_pvt.rsb_m3m3 = rsb_m3m3
        class_pvt.rp_m3m3 = rp_m3m3
        class_pvt.pb_atma = pb_atma
        class_pvt.tres_C = tres_C
        class_pvt.bob_m3m3 = bob_m3m3
        class_pvt.muob_cP = muob_cP
        class_pvt.bwSC_m3m3 = bwSC_m3m3
        class_pvt.qgas_free_sm3day = qgas_free_sm3day
        class_pvt.salinity_ppm = salinity_ppm
        class_pvt.calc(Pintake_, tksep_C, Qliq_, Fw_ / 100, z_class, pseudo_class)

        self.assertEqual(64.28618258473426, class_pvt.q_mix_rc_m3day)

    def test_2nd_calc_standing(self):
        class_pvt = standing.calc_standing()
        gamma_oil = 0.82
        gamma_wat = 1
        gamma_gas = 0.796
        rsb_m3m3 = 15.6
        rp_m3m3 = 7.071955128205128
        pb_atma = 139
        tres_C = 89
        bob_m3m3 = 1.278
        muob_cP = 7.76 # Вязкость нефти, из PVT
        qgas_free_sm3day = 0
        tksep_C = 90
        bwSC_m3m3 = gamma_wat / 1
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)

        Pintake_ = 34.666
        Qliq_ = 62.4
        Fw_ = 94.4
        ksep_fr = 0.7
        Tintake_ = 90
        z_class = calc_func.z_factor_dran_chuk()
        pseudo_class = calc_func.pseudo_standing()

        class_pvt.gamma_gas = gamma_gas
        class_pvt.gamma_oil = gamma_oil
        class_pvt.gamma_wat = gamma_wat
        class_pvt.ksep_fr = ksep_fr
        class_pvt.rsb_m3m3 = rsb_m3m3
        class_pvt.rp_m3m3 = rp_m3m3
        class_pvt.pb_atma = pb_atma
        class_pvt.tres_C = tres_C
        class_pvt.bob_m3m3 = bob_m3m3
        class_pvt.muob_cP = muob_cP
        class_pvt.bwSC_m3m3 = bwSC_m3m3
        class_pvt.salinity_ppm = salinity_ppm
        class_pvt.qgas_free_sm3day = qgas_free_sm3day
        class_pvt.calc(Pintake_, tksep_C, Qliq_, Fw_ / 100, z_class, pseudo_class)

        self.assertEqual(65.68000526272196, class_pvt.q_mix_rc_m3day)

    def test_calc_cain(self):
        class_pvt = mc_cain.calc_cain()
        gamma_oil = 0.777
        gamma_wat = 1.054
        gamma_gas = 0.749
        rsb_m3m3 = 2.20773800714468
        rp_m3m3 = 2.20773800714468
        pb_atma = 106.648752781247
        tres_C = 101.5
        bob_m3m3 = 1.52086750956118
        muob_cP = 5.58 # Вязкость нефти, из PVT
        qgas_free_sm3day = 0
        tksep_C = 90
        bwSC_m3m3 = gamma_wat / 1
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)

        Pintake_ = 34.5
        Qliq_ = 61.2
        Fw_ = 96.7
        ksep_fr = 0.7
        Tintake_ = 90

        z_class = calc_func.z_factor_2015_kareem()
        pseudo_class = calc_func.pseudo_standing()

        class_pvt.gamma_gas = gamma_gas
        class_pvt.gamma_oil = gamma_oil
        class_pvt.gamma_wat = gamma_wat
        class_pvt.ksep_fr = ksep_fr
        class_pvt.rsb_m3m3 = rsb_m3m3
        class_pvt.rp_m3m3 = rp_m3m3
        class_pvt.pb_atma = pb_atma
        class_pvt.tres_C = tres_C
        class_pvt.bob_m3m3 = bob_m3m3
        class_pvt.muob_cP = muob_cP
        class_pvt.bwSC_m3m3 = bwSC_m3m3
        class_pvt.salinity_ppm = salinity_ppm
        class_pvt.qgas_free_sm3day = qgas_free_sm3day
        class_pvt.calc(Pintake_, tksep_C, Qliq_, Fw_ / 100, z_class, pseudo_class)

        self.assertEqual(64.2937165646473749, class_pvt.q_mix_rc_m3day)

    def test_calc_straight_line(self):
        class_pvt = straight_line.calc_straight_line()
        gamma_oil = 0.777
        gamma_wat = 1.054
        gamma_gas = 0.749
        rsb_m3m3 = 2.38271728395062
        rp_m3m3 = 2.38271728395062
        pb_atma = 97.05
        tres_C = 101.5
        bob_m3m3 = 1.23099628099174
        muob_cP = 5.58 # Вязкость нефти, из PVT
        qgas_free_sm3day = 0
        tksep_C = 90
        bwSC_m3m3 = gamma_wat / 1
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)

        Pintake_ = 34.5
        Qliq_ = 61.2
        Fw_ = 96.7
        Tintake_ = 90
        ksep_fr = 0.7
        z_class = calc_func.z_factor_2015_kareem()
        pseudo_class = calc_func.pseudo_standing()

        class_pvt.gamma_gas = gamma_gas
        class_pvt.gamma_oil = gamma_oil
        class_pvt.gamma_wat = gamma_wat
        class_pvt.ksep_fr = ksep_fr
        class_pvt.rsb_m3m3 = rsb_m3m3
        class_pvt.rp_m3m3 = rp_m3m3
        class_pvt.pb_atma = pb_atma
        class_pvt.tres_C = tres_C
        class_pvt.bob_m3m3 = bob_m3m3
        class_pvt.muob_cP = muob_cP
        class_pvt.bwSC_m3m3 = bwSC_m3m3
        class_pvt.salinity_ppm = salinity_ppm
        class_pvt.qgas_free_sm3day = qgas_free_sm3day
        class_pvt.calc(Pintake_, tksep_C, Qliq_, Fw_ / 100, z_class, pseudo_class)

        self.assertEqual(63.595725611969991, class_pvt.q_mix_rc_m3day)

    def test_2nd_calc_straight_line(self):
        class_pvt = straight_line.calc_straight_line()
        gamma_oil = 0.836
        gamma_wat = 1.17
        gamma_gas = 0.7
        rsb_m3m3 = 178.737
        rp_m3m3 = 214.286
        pb_atma = 206
        tres_C = 101.5
        bob_m3m3 = -1
        muob_cP = 0.98 # Вязкость нефти, из PVT
        qgas_free_sm3day = 5000
        tksep_C = 20.0879398496241
        bwSC_m3m3 = gamma_wat / 1
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)

        Pintake_ = 52.0462549043612
        Qliq_ = 14
        Fw_ = 0
        Tintake_ = 90
        ksep_fr = 0.7
        z_class = calc_func.z_factor()
        pseudo_class = calc_func.pseudo()

        class_pvt.gamma_gas = gamma_gas
        class_pvt.gamma_oil = gamma_oil
        class_pvt.gamma_wat = gamma_wat
        class_pvt.ksep_fr = ksep_fr
        class_pvt.rsb_m3m3 = rsb_m3m3
        class_pvt.rp_m3m3 = rp_m3m3
        class_pvt.pb_atma = pb_atma
        class_pvt.tres_C = tres_C
        class_pvt.bob_m3m3 = bob_m3m3
        class_pvt.muob_cP = muob_cP
        class_pvt.bwSC_m3m3 = bwSC_m3m3
        class_pvt.salinity_ppm = salinity_ppm
        class_pvt.qgas_free_sm3day = qgas_free_sm3day
        class_pvt.calc(Pintake_, tksep_C, Qliq_, Fw_ / 100, z_class, pseudo_class)

        self.assertAlmostEqual(149.716252013342, class_pvt.q_mix_rc_m3day, places = 3),

# 149.71625201334226 != 149.716252013342
# Expected :149.716252013342
# Actual   :149.71625201334226


if __name__ == '__main__':
    unittest.main()
