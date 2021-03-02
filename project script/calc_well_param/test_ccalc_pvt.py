import unittest
import ccalc_pvt

class ccalc_pvt_test(unittest.TestCase):
    def test_calc_pvt(self):
        p_pvt = 17
        t_pvt = 290.69342105263157894736842105
        r_sbcalc = 75.6
        gamma_o = 0.84
        gamma_g = 0.8
        yukos_standard = 0
        p_rbcalc = -1
        b_ro = -1
        salinitycalc = 0
        gamma_w = 1
        (p_b, r_s, b_o, mu_o, z, b_g, mu_g, b_w, mu_w) = ccalc_pvt.calc_pvt(p_pvt, t_pvt, r_sbcalc, gamma_o, gamma_g, yukos_standard, p_rbcalc, b_ro, salinitycalc, gamma_w)
        self.assertEqual(round(p_b, 5), round(88.68742580342252154701730802, 5))
        self.assertEqual(round(r_s, 5), round(10.28665195909544, 5))
        self.assertEqual(round(b_o, 5), round(1.018997399542232496, 5))
        self.assertEqual(round(mu_o, 5), round(18.878011665268662137588884286, 5))
        self.assertEqual(round(z, 5), round(0.93082275390625, 5))
        self.assertEqual(round(b_g, 5), round(0.0545567586628556999500877142, 5))
        self.assertEqual(round(mu_g, 5), round(0.0103320541209641420522244070, 5))
        self.assertEqual(round(b_w, 5), round(1.0005032035921262253403000912, 5))
        self.assertEqual(round(mu_w, 5), round(1.0561330030105375326607738572, 5))

    def test_water_fvf(self):
        oil_pressure_in_m_pa = 1.72210
        t = 290.69342105263157894736842105
        result = ccalc_pvt.water_fvf(oil_pressure_in_m_pa, t)
        self.assertEqual(round(result, 5), round(1.0005032035921262253403000912, 5))

    def test_water_viscosity(self):
        oil_pressure_in_m_pa = 1.72210
        t = 290.69342105263157894736842105
        s = 0
        gammaW = 1
        result = ccalc_pvt.water_viscosity(oil_pressure_in_m_pa, t, s, gammaW)
        self.assertEqual(round(result, 5), round(1.0561330030105375326607738572, 5))

    def test_z_factor_dran_chuk(self):
        tPr = 1.2483699151614525348768540068
        pPr = 0.3814276348430080090722234035
#       result = ccalc_pvt.z_factor_dran_chuk(tPr, pPr) - вызов до ввода классов, ошибочный - такой функции нет
        import calc_func
        args_dic = {}
        args_dic['t_pseudo'] = tPr
        args_dic['p_pseudo'] = pPr
        z_class = calc_func.z_factor_dran_chuk()
        result = z_class.z_factor(args_dic)
        self.assertEqual(round(result, 5), round(0.93082275390625, 5))

    def test_gas_fvf(self):
        oil_pressure_in_m_pa = 1.72210
        t = 290.69342105263157894736842105
        zi = 0.93082275390625
        result = ccalc_pvt.gas_fvf(t, oil_pressure_in_m_pa, zi)
        self.assertEqual(round(result, 5), round(0.0545567586628556999500877142, 5))

    def test_g_visc(self):
        oil_pressure_in_m_pa = 1.72210
        t = 290.69342105263157894736842105
        zi = 0.93082275390625
        gammaG = 0.8
        result = ccalc_pvt.g_visc(t, oil_pressure_in_m_pa, zi, gammaG)
        self.assertEqual(round(result, 5), round(0.0103320541209641420522244070, 5))

    def test_dead_oil_viscosity_beggs_robin(self):
        t = 290.69342105263157894736842105
        gammao = 0.84
        result = ccalc_pvt.dead_oil_viscosity_beggs_robin(t, gammao)
        self.assertEqual(round(result, 5), round(34.516658647383, 5))

    def test_saturated_oil_viscosity_beggs_r(self):
        rSb = 75.6
        deadOilViscosity = 34.516658647383
        result = ccalc_pvt.saturated_oil_viscosity_beggs_r(rSb, deadOilViscosity)
        self.assertEqual(round(result, 5), round(4.0401170805367899939907315272, 5))

    def test_bubble_point_standing(self):
        rSb = 75.6
        t = 290.69342105263157894736842105
        gammaG = 0.8
        gammaO = 0.84
        result = ccalc_pvt.bubble_point_standing(rSb, gammaG, t, gammaO)
        self.assertEqual(round(result, 5), round(8.984036233886701432712853302, 5))

    def test_gor_standing(self):
        oil_pressure_in_m_pa = 1.72210
        t = 290.69342105263157894736842105
        gammaG = 0.8
        gammaO = 0.84
        result = ccalc_pvt.gor_standing(oil_pressure_in_m_pa, gammaG, t, gammaO)
        self.assertEqual(round(result, 5), round(10.28665195909544, 5))

    def test_fvf_saturated_oil_standing(self):
        gorVelardesi = 10.28665195909544
        gammaG = 0.8
        gammaO = 0.84
        t = 290.69342105263157894736842105
        result = ccalc_pvt.fvf_saturated_oil_standing(gorVelardesi, gammaG, t, gammaO)
        self.assertEqual(round(result, 5), round(1.018997399542232496, 5))

    def test_calc_ST(self):
        p_atma = 181.032556392203
        t_C = 90
        waterСut = 0.944
        PVTdic = {}
        PVTdic['gamma_oil'] = 0.82
        rPVTdic = ccalc_pvt.calc_ST(p_atma, t_C, waterСut, PVTdic)
        self.assertAlmostEqual(40.7896986418018, rPVTdic['ST_liqgas_dyncm'], places=8)
        self.assertAlmostEqual(11.0039306073889, rPVTdic['ST_oilgas_dyncm'], places=8)
        self.assertAlmostEqual(42.5566509828263, rPVTdic['ST_watgas_dyncm'], places=8)

if __name__=='__main__':
    unittest.main()
