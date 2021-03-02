import unittest
import calc_q_mix
import ccalc_pvt
import calc_func
from CPVT.standing import calc_standing

class QmixTestCase(unittest.TestCase):
    def test_calc_q_mix(self):
        gamma_oil = 0.777
        gamma_wat = 1.054
        gamma_gas = 0.749
        rsb_m3m3 = 5.966
        rp_m3m3 = float(5.966)
        pb_atma = 243
        tres_C = 101.5
        bob_m3m3 = 1.582
        t_ksep_C = 90
        muob_cP = 5.580
        ksep_fr = 0.7
        qgas_free_sm3day = 0
        bwSC_m3m3 = gamma_wat / 1
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)
        class_pvt = calc_standing()
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

        Pintake_ = 34.5
        Qliq_ = 61.2
        Fw_ = 96.7
        Tintake_ = 90

        class_qmix = calc_q_mix.Qmix(Pintake_, Tintake_, Qliq_, Fw_ / 100, class_pvt, calc_func.z_class_select,
                                       calc_func.pseudo_class_select)
        class_qmix.calc_q_mix_rc_m3day()
        self.assertAlmostEqual(64.28618258473425, class_qmix.q_mix_rc_m3day, places=5)  # 64.28618258473425 != 64.28618258473426

    def test_2nd_calc_q_mix(self):
        gamma_oil = 0.777
        gamma_wat = 1.054
        gamma_gas = 0.749
        rsb_m3m3 = 267
        rp_m3m3 = float(7.072)
        pb_atma = 243
        tres_C = 101.5
        bob_m3m3 = 1.582
        t_ksep_C = 90
        muob_cP = 5.580
        ksep_fr = 0.7
        qgas_free_sm3day = 0
        bwSC_m3m3 = gamma_wat / 1
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)
        class_pvt = calc_standing()
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

        Pintake_ = 34.667
        Qliq_ = 62.4
        Fw_ = 94.4
        Tintake_ = 90
        class_qmix = calc_q_mix.Qmix(Pintake_, Tintake_, Qliq_, Fw_ / 100, class_pvt, calc_func.z_class_select,
                                     calc_func.pseudo_class_select)
        class_qmix.calc_q_mix_rc_m3day()

        self.assertAlmostEqual(66.17282973608943, class_qmix.q_mix_rc_m3day, places=5)  #  66.17282659245636 != 66.17282973608943






if __name__ == '__main__':
    unittest.main()

