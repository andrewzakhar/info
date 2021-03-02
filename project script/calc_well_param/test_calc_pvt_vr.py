import os, sys
sys.path.append(os.path.join(os.path.abspath(''), '..'))
sys.path.append(os.path.join(os.path.abspath(''), '..', 'common'))

import ccalc_pvt
import calc_func

import unittest

class unf_calc_Sal_BwSC_ppm_test_case(unittest.TestCase):
    def test_unf_calc_Sal_BwSC_ppm(self):
        PVTdic = {}
        PVTdic['bwSC_m3m3'] = 1
        PVTdic['salinity_ppm'] = ccalc_pvt.unf_calc_Sal_BwSC_ppm(PVTdic['bwSC_m3m3'])
        self.assertEqual(1363.1482481105195, PVTdic['salinity_ppm'])


class ccalc_pvt_test_case(unittest.TestCase):
    def test_calc_pvt_vr(self):
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
        PVTdic['gamma_gas'] = float(gamma_gas_)
        PVTdic['gamma_oil'] = float(gamma_oil_)
        PVTdic['gamma_wat'] = float(gamma_wat_)
        PVTdic['rsb_m3m3'] = float(Rsb_)
        PVTdic['pb_atma'] = float(Pb_)
        PVTdic['tres_C'] = float(Tres_)
        PVTdic['bob_m3m3'] = float(Bob_)
        PVTdic['muob_cP'] = float(mu_)
        PVTdic['qgas_free_sm3day'] = float(0)
        PVTdic['rp_m3m3'] = float(7.071955128205128)
        PVTdic['tksep_C'] = float(90)

        PVTdic['bwSC_m3m3'] = PVTdic['gamma_wat'] / 1
        PVTdic['salinity_ppm'] = ccalc_pvt.unf_calc_Sal_BwSC_ppm(PVTdic['bwSC_m3m3'])

        Pintake_ = 34.666
        Qliq_ = 62.4
        Fw_ = 94.4
        Tintake_ = 90
        z_class = calc_func.z_factor_dran_chuk()
        z = z_class.z_factor
        PVTdic = ccalc_pvt.calc_pvt_vr(Pintake_, PVTdic['tksep_C'], Qliq_, Fw_ / 100,
                                       PVTdic['PVTcorr'], PVTdic, z)
# было  PVTdic = ccalc_pvt.calc_pvt_vr(Pintake_, PVTdic['tksep_C'], Qliq_, Fw_ / 100,
#                                      PVTdic['PVTcorr'], PVTdic, calc_func.z_factor_dran_chuk)
        qOil, qWat, qGas = ccalc_pvt.Q_fluid_calc(Qliq_, Fw_ / 100, PVTdic['bo_m3m3'], PVTdic['bw_m3m3'],
                                                  PVTdic['rp_m3m3'],  PVTdic['qgas_free_sm3day'],
                                                  PVTdic['rs_m3m3'], PVTdic['bg_m3m3'])
        Q_mix_intake_test = None
        if (qOil != qOil or qWat != qWat or qGas != qGas) == False:
            Q_mix_intake_test = qOil + qWat + qGas
        self.assertAlmostEqual(65.53034385597518, Q_mix_intake_test, places=0)
        
if __name__=='__main__':
    unittest.main()
# 65.67311820531015 != 65.53034385597518
# Expected :65.53034385597518
# Actual   :65.67311820531015
