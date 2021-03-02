
import unittest
from CPIPE.pipe import CPipe
import ccalc_pvt
from CPVT.standing import calc_standing


class Test_ModSeparationPipe(unittest.TestCase):
    def test_pipe_atma_modern(self):
        # Вынгаяхинское месторождение
        gamma_oil = 0.777
        gamma_wat = 1.054
        gamma_gas = 0.749
        rsb_m3m3 = 267
        rp_m3m3 = float(7.071955128205128)
        pb_atma = 243
        pksep_atma = 34.666666666666664
        tres_C = 101.5
        bob_m3m3 = 1.582
        muob_cP = 5.58  # Вязкость нефти, из PVT
        bwSC_m3m3 = gamma_wat / 1
        PVTCorr = 0
        ksep_fr = 0.7
        qgas_free_sm3day = 0
        tksep_C = 90
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)
        class_pvt = calc_standing()
        class_pvt.PVTCorr = PVTCorr
        class_pvt.gamma_oil = gamma_oil
        class_pvt.gamma_wat = gamma_wat
        class_pvt.gamma_gas = gamma_gas
        class_pvt.rsb_m3m3 = rsb_m3m3
        class_pvt.rp_m3m3 = rp_m3m3
        class_pvt.pb_atma = pb_atma
        class_pvt.pksep_atma = pksep_atma
        class_pvt.tres_C = tres_C
        class_pvt.bob_m3m3 = bob_m3m3
        class_pvt.tksep_C = tksep_C
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
        classPIPE = CPipe()
        pressure = classPIPE.pipe_atma_modern(Qliq_, Fw_, Length_, Pcalc_, Calc_along_flow_, class_pvt, Theta_deg,
                                                 Dtub_, Hydr_corr_, Tcalc_, Tother_)
        self.assertAlmostEqual(2.46295286730971, pressure, places=6)

    def test_pipe_atma_modern_2(self):
        # Вынгаяхинское месторождение
        gamma_oil = 0.777
        gamma_wat = 1.054
        gamma_gas = 0.749
        rsb_m3m3 = 267
        rp_m3m3 = float(7.071955128205128)
        pb_atma = 243
        pksep_atma = 34.666666666666664
        tres_C = 101.5
        bob_m3m3 = 1.582
        muob_cP = 5.58  # Вязкость нефти, из PVT
        bwSC_m3m3 = gamma_wat / 1
        PVTCorr = 0
        ksep_fr = 0.7
        qgas_free_sm3day = 0
        tksep_C = 90
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)
        class_pvt = calc_standing()
        class_pvt.PVTCorr = PVTCorr
        class_pvt.gamma_oil = gamma_oil
        class_pvt.gamma_wat = gamma_wat
        class_pvt.gamma_gas = gamma_gas
        class_pvt.rsb_m3m3 = rsb_m3m3
        class_pvt.rp_m3m3 = rp_m3m3
        class_pvt.pb_atma = pb_atma
        class_pvt.pksep_atma = pksep_atma
        class_pvt.tres_C = tres_C
        class_pvt.bob_m3m3 = bob_m3m3
        class_pvt.tksep_C = tksep_C
        class_pvt.muob_cP = muob_cP  # Вязкость нефти, из PVT
        class_pvt.qgas_free_sm3day = qgas_free_sm3day
        class_pvt.ksep_fr = ksep_fr
        class_pvt.bwSC_m3m3 = bwSC_m3m3
        class_pvt.salinity_ppm = salinity_ppm

        Qliq_ = 62.4
        Fw_ = 94.4
        Length_ = 10.0
        Pcalc_ = 2.46295286730971
        Calc_along_flow_ = 0
        Theta_deg = 89.89
        Dtub_ = 79
        Hydr_corr_ = 0
        Tcalc_ = 101.5
        Tother_ = 101.5
        classPIPE = CPipe()
        pressure = classPIPE.pipe_atma_modern(Qliq_, Fw_, Length_, Pcalc_, Calc_along_flow_, class_pvt, Theta_deg,
                                              Dtub_, Hydr_corr_, Tcalc_, Tother_)
        self.assertAlmostEqual(3.3980311137350356, pressure, places=6)

    def test_pipe_atma_modern_3(self):
        # Вынгаяхинское месторождение
        gamma_oil = 0.777
        gamma_wat = 1.054
        gamma_gas = 0.749
        rsb_m3m3 = 267
        rp_m3m3 = float(7.071955128205128)
        pb_atma = 243
        pksep_atma = 34.666666666666664
        tres_C = 101.5
        bob_m3m3 = 1.582
        muob_cP = 5.58  # Вязкость нефти, из PVT
        bwSC_m3m3 = gamma_wat / 1
        PVTCorr = 0
        ksep_fr = 0.7
        qgas_free_sm3day = 0
        tksep_C = 90
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)
        class_pvt = calc_standing()
        class_pvt.PVTCorr = PVTCorr
        class_pvt.gamma_oil = gamma_oil
        class_pvt.gamma_wat = gamma_wat
        class_pvt.gamma_gas = gamma_gas
        class_pvt.rsb_m3m3 = rsb_m3m3
        class_pvt.rp_m3m3 = rp_m3m3
        class_pvt.pb_atma = pb_atma
        class_pvt.pksep_atma = pksep_atma
        class_pvt.tres_C = tres_C
        class_pvt.bob_m3m3 = bob_m3m3
        class_pvt.tksep_C = tksep_C
        class_pvt.muob_cP = muob_cP  # Вязкость нефти, из PVT
        class_pvt.qgas_free_sm3day = qgas_free_sm3day
        class_pvt.ksep_fr = ksep_fr
        class_pvt.bwSC_m3m3 = bwSC_m3m3
        class_pvt.salinity_ppm = salinity_ppm

        Qliq_ = 62.4
        Fw_ = 94.4
        Length_ = 10.0
        Pcalc_ = 3.3980311137350356
        Calc_along_flow_ = 0
        Theta_deg = 89.84
        Dtub_ = 79
        Hydr_corr_ = 0
        Tcalc_ = 101.5
        Tother_ = 101.5
        classPIPE = CPipe()
        pressure = classPIPE.pipe_atma_modern(Qliq_, Fw_, Length_, Pcalc_, Calc_along_flow_, class_pvt, Theta_deg,
                                              Dtub_, Hydr_corr_, Tcalc_, Tother_)
        self.assertAlmostEqual(4.337339108338191, pressure, places=6)

    def test_pipe_atma_modern_4(self):
        # Вынгаяхинское месторождение
        gamma_oil = 0.777
        gamma_wat = 1.054
        gamma_gas = 0.749
        rsb_m3m3 = 267
        rp_m3m3 = float(7.071955128205128)
        pb_atma = 243
        pksep_atma = 34.666666666666664
        tres_C = 101.5
        bob_m3m3 = 1.582
        muob_cP = 5.58  # Вязкость нефти, из PVT
        bwSC_m3m3 = gamma_wat / 1
        PVTCorr = 0
        ksep_fr = 0.7
        qgas_free_sm3day = 0
        tksep_C = 90
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)
        class_pvt = calc_standing()
        class_pvt.PVTCorr = PVTCorr
        class_pvt.gamma_oil = gamma_oil
        class_pvt.gamma_wat = gamma_wat
        class_pvt.gamma_gas = gamma_gas
        class_pvt.rsb_m3m3 = rsb_m3m3
        class_pvt.rp_m3m3 = rp_m3m3
        class_pvt.pb_atma = pb_atma
        class_pvt.pksep_atma = pksep_atma
        class_pvt.tres_C = tres_C
        class_pvt.bob_m3m3 = bob_m3m3
        class_pvt.tksep_C = tksep_C
        class_pvt.muob_cP = muob_cP  # Вязкость нефти, из PVT
        class_pvt.qgas_free_sm3day = qgas_free_sm3day
        class_pvt.ksep_fr = ksep_fr
        class_pvt.bwSC_m3m3 = bwSC_m3m3
        class_pvt.salinity_ppm = salinity_ppm

        Qliq_ = 62.4
        Fw_ = 94.4
        Length_ = 10.0
        Pcalc_ = 4.337339108338191
        Calc_along_flow_ = 0
        Theta_deg = 89.79
        Dtub_ = 79
        Hydr_corr_ = 0
        Tcalc_ = 101.5
        Tother_ = 101.5
        classPIPE = CPipe()
        pressure = classPIPE.pipe_atma_modern(Qliq_, Fw_, Length_, Pcalc_, Calc_along_flow_, class_pvt, Theta_deg,
                                              Dtub_, Hydr_corr_, Tcalc_, Tother_)
        self.assertAlmostEqual(5.279242958725155, pressure, places=6)

    def test_pipe_atma_modern_5(self):
        # Вынгаяхинское месторождение
        gamma_oil = 0.777
        gamma_wat = 1.054
        gamma_gas = 0.749
        rsb_m3m3 = 267
        rp_m3m3 = float(7.071955128205128)
        pb_atma = 243
        pksep_atma = 34.666666666666664
        tres_C = 101.5
        bob_m3m3 = 1.582
        muob_cP = 5.58  # Вязкость нефти, из PVT
        bwSC_m3m3 = gamma_wat / 1
        PVTCorr = 0
        ksep_fr = 0.7
        qgas_free_sm3day = 0
        tksep_C = 90
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)
        class_pvt = calc_standing()
        class_pvt.PVTCorr = PVTCorr
        class_pvt.gamma_oil = gamma_oil
        class_pvt.gamma_wat = gamma_wat
        class_pvt.gamma_gas = gamma_gas
        class_pvt.rsb_m3m3 = rsb_m3m3
        class_pvt.rp_m3m3 = rp_m3m3
        class_pvt.pb_atma = pb_atma
        class_pvt.pksep_atma = pksep_atma
        class_pvt.tres_C = tres_C
        class_pvt.bob_m3m3 = bob_m3m3
        class_pvt.tksep_C = tksep_C
        class_pvt.muob_cP = muob_cP  # Вязкость нефти, из PVT
        class_pvt.qgas_free_sm3day = qgas_free_sm3day
        class_pvt.ksep_fr = ksep_fr
        class_pvt.bwSC_m3m3 = bwSC_m3m3
        class_pvt.salinity_ppm = salinity_ppm

        Qliq_ = 62.4
        Fw_ = 94.4
        Length_ = 10.0
        Pcalc_ = 5.279242958725155
        Calc_along_flow_ = 0
        Theta_deg = 89.73
        Dtub_ = 79
        Hydr_corr_ = 0
        Tcalc_ = 101.5
        Tother_ = 101.5
        classPIPE = CPipe()
        pressure = classPIPE.pipe_atma_modern(Qliq_, Fw_, Length_, Pcalc_, Calc_along_flow_, class_pvt, Theta_deg,
                                              Dtub_, Hydr_corr_, Tcalc_, Tother_)
        self.assertAlmostEqual(6.222901787522267, pressure, places=6)

if __name__ == '__main__':
    unittest.main()