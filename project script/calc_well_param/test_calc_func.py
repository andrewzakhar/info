
import unittest
import calc_func
import ccalc_pvt
import calc_types
from CPVT import standing

class z_factor_2015_kareem_test_case(unittest.TestCase):
    def test_z_factor_2015_kareem(self):
        args_dic = {}
        args_dic['t_pseudo'] = 1.56342394263404
        args_dic['p_pseudo'] = 0.777797816358698
        z_class = calc_func.z_factor_2015_kareem()
        z = z_class.z_factor(args_dic)
        self.assertEqual(0.937556584557303, z)

class z_factor_dran_chuk_test_case(unittest.TestCase):
    def test_z_factor_dran_chuk(self):
        args_dic = {}
        args_dic['t_pseudo'] = 1.56342394263404
        args_dic['p_pseudo'] = 0.777797816358698
        z_class = calc_func.z_factor_dran_chuk()
        z = z_class.z_factor(args_dic)
        self.assertAlmostEqual(0.934519684314728, z, places = 3)

class ODESolverInit_test_case(unittest.TestCase):
    def test_ODESolverInit(self):
        #  метод расчета 0
        solverType = 0
        M = 2
        h = 10
        y0_arr = [1.536]
        eps = 0.1
        #  размер системы  - одна пара переменных - давление и температура
        N = 1
        X_arr = [0, 10]

        state = calc_func.ODESolverInit(solverType, y0_arr, N, X_arr, M, eps, h)

        # expected values
        e = calc_func.State()
        e.rState.Stage = -1
        e.rState.IA = [0]*6
        e.rState.BA = [False]
        e.rState.CA = None
        e.rState.RA = [0]*6
        e.DY = None
        e.Eps = 0.1
        e.EScale = None
        e.FracEps = False
        e.H = 10
        e.M = 2
        e.N = 1
        e.RepNFEV = None
        e.RepTerminationType = 0
        e.RKA = None
        e.RKB = None
        e.RKC = None
        e.RKCS = None
        e.solverType = 0
        e.X = None
        e.XG = [0, 10]
        e.XScale= 1
        e.y = None
        e.YC = [1.536]
        e.YN = None
        e.YNS = None
        e.YTbl = None

        self.assertEqual(e.rState.Stage, state.rState.Stage)
        self.assertEqual(e.rState.IA, state.rState.IA)
        self.assertEqual(e.rState.BA, state.rState.BA)
        self.assertEqual(e.rState.CA, state.rState.CA)
        self.assertEqual(e.rState.RA, state.rState.RA)
        self.assertEqual(e.DY, state.DY)
        self.assertEqual(e.Eps, state.Eps)
        self.assertEqual(e.EScale, state.EScale)
        self.assertEqual(e.FracEps, state.FracEps)
        self.assertEqual(e.H, state.H)
        self.assertEqual(e.M, state.M)
        self.assertEqual(e.N, state.N)
        self.assertEqual(e.RepNFEV, state.RepNFEV)
        self.assertEqual(e.RepTerminationType, state.RepTerminationType)
        self.assertEqual(e.RKA, state.RKA)
        self.assertEqual(e.RKB, state.RKB)
        self.assertEqual(e.RKC, state.RKC)
        self.assertEqual(e.RKCS, state.RKCS)
        self.assertEqual(e.solverType, state.solverType)
        self.assertEqual(e.X, state.X)
        self.assertEqual(e.XG, state.XG)
        self.assertEqual(e.XScale, state.XScale)
        self.assertEqual(e.y, state.y)
        self.assertEqual(e.YC, state.YC)
        self.assertEqual(e.YN, state.YN)
        self.assertEqual(e.YNS, state.YNS)
        self.assertEqual(e.YTbl, state.YTbl)

class h_l_arr_theta_deg_test_case(unittest.TestCase):
    def test_h_l_arr_theta_deg(self):
        flow_pattern = 1
        lambda_l = 0.146858249840252
        n_fr = 3.33342096919987
        n_lv = 1.38982374517846
        arr_theta_deg = 89.9500000000073
        Payne_et_all = 0
        Hl_out_fr = calc_func.h_l_arr_theta_deg(flow_pattern, lambda_l, n_fr, n_lv, arr_theta_deg, Payne_et_all)
        self.assertEqual(0.3346469958123525, Hl_out_fr)

class calc_friction_factor_test_case(unittest.TestCase):
    def test_calc_friction_factor(self):
        n_re = 15570.3282123982
        roughness_d = 0.00126582278481013
        Payne_et_all_friction = 1
        f_n = calc_func.calc_friction_factor(n_re, roughness_d, Payne_et_all_friction)
        self.assertEqual(0.029891633467844826, f_n)

class BegsBrillGradient_test_case(unittest.TestCase):
    def test_BegsBrillGradient(self):
        arr_d_m = 0.079
        arr_theta_deg = 89.9500000000073
        eps_m = 0.0001
        Ql_rc_m3day = 65.2660317740851
        Qg_rc_m3day = 8.48031982336487
        Mul_rc_cP = 0.837091367843072
        Mug_rc_cP = 0.0124383986806129
        sigma_l_Nm = 0.0640952157612172
        rho_lrc_kgm3 = 949.132237730242
        rho_grc_kgm3 = 1.20723236536855
        Payne_et_all_holdup = 0
        Payne_et_all_friction = 1
        c_calibr_grav = 1
        c_calibr_fric = 1
        dp_dl_arr = calc_func.BegsBrillGradient(arr_d_m, arr_theta_deg, eps_m, Ql_rc_m3day, Qg_rc_m3day, Mul_rc_cP, Mug_rc_cP, sigma_l_Nm, rho_lrc_kgm3, rho_grc_kgm3, Payne_et_all_holdup=0, Payne_et_all_friction=1, c_calibr_grav=1, c_calibr_fric=1)

        e = [0] * 8
        e[0] = 0.0885311272461878
        e[1] = 0.0885311272461878
        e[2] = 0.0000599214320492055
        e[3] = 0
        e[4] = 0.15410949653187
        e[5] = 0.0200241654484489
        e[6] = 0.963138504715659
        e[7] = 3
        i = 0
        for val in e:
            s = f"i = {i}"
            self.assertAlmostEqual(val, dp_dl_arr[i],  places=3, msg=s)
            print(i, val, dp_dl_arr[i])
            i += 1

class calc_grad_test_case(unittest.TestCase):
    def test_calc_grad(self):
        l_m = 0
        p_atma = 1.536
        t_C = 89
        calc_dtdl = False
        pcas_atma = 0.95
        theta_sign = 0
        d_m = 0.079
        theta_deg = 89.95
        rough_m = 0.0001
        Hv_m = 0
        angle_hmes_deg = 89.9500000000073

        Qliq = 62.4
        Fw = 94.4
        #    elif oilfield_ == 'Вынгаяхинское':
        PVTdic = {}
        PVTdic['PVTcorr'] = 0
        PVTdic['ksep_fr'] = 0.7  # Коэффициент сепарации
        PVTdic['qgas_free_sm3day'] = 0

        PVTdic['gamma_gas'] = 0.796
        PVTdic['gamma_oil'] = 0.82
        PVTdic['gamma_wat'] = 1
        PVTdic['rsb_m3m3'] = 15.6
        PVTdic['rp_m3m3'] = float(7.071955128205128) #Неизвестно где взять, входные данные утеряны
        PVTdic['pb_atma'] = 139
        PVTdic['tres_C'] = 89
        PVTdic['bob_m3m3'] = 1.278
        PVTdic['muob_cP'] = 7.76  # Вязкость нефти, из PVT
        PVTdic['bwSC_m3m3'] = PVTdic['gamma_wat'] / 1
        PVTdic['salinity_ppm'] = ccalc_pvt.unf_calc_Sal_BwSC_ppm(PVTdic['bwSC_m3m3'])
        PVTdic['qliq_rc_m3day'] = 65.2660317740851
        PVTdic['q_gas_rc_m3day'] = 8.48031982336487
        PVTdic['mu_liq_cP'] = 0.837091367843072
        PVTdic['mu_gas_cP'] = 0.0124383986806129
        PVTdic['sigma_liq_Nm'] = 0.0640952157612172
        PVTdic['rho_liq_rc_kgm3'] = 949.132237730242
        PVTdic['rho_gas_rc_kgm3'] = 1.20723236536855

        pCalc = calc_types.paramCalc(aCorrelation=PVTdic['PVTcorr'], aCalcAlongCoord=False, aFlowAlongCoord=False)

        dp_dl = calc_func.calc_grad(l_m, p_atma, t_C, pCalc, d_m, theta_deg, angle_hmes_deg, rough_m, Qliq, Fw, PVTdic,
                                    calc_dtdl)
        self.assertAlmostEqual(0.088591048678237, dp_dl, places=8)
# 0.088591048678237 != 0.08622400146927559

# 0.088591048678237 != 0.08624538722427744

    def test_2calc_grad_class_pvt(self):
        #    elif oilfield_ == 'Вынгаяхинское':
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
        class_pvt.mu_liq_cP = 0.837091367843072
        class_pvt.mu_gas_cP = 0.0124383986806129
        class_pvt.sigma_liq_Nm = 0.0640952157612172
        class_pvt.rho_liq_rc_kgm3 = 949.132237730242
        class_pvt.rho_gas_rc_kgm3 = 1.20723236536855
        class_pvt.qliq_rc_m3day = 65.2660317740851
        class_pvt.q_gas_rc_m3day = 8.48031982336487

        l_m = 0
        p_atma = 1.536
        t_C = 89
        calc_dtdl = False
        pcas_atma = 0.95
        theta_sign = 0
        d_m = 0.079
        theta_deg = 89.95
        rough_m = 0.0001
        Hv_m = 0
        angle_hmes_deg = 89.9500000000073

        Qliq = 62.4
        Fw = 94.4
        pCalc = calc_types.paramCalc(aCorrelation=class_pvt.PVTCorr, aCalcAlongCoord=False, aFlowAlongCoord=False)

        dp_dl = calc_func.calc_grad_pvtcls(l_m, p_atma, t_C, pCalc, d_m, theta_deg, angle_hmes_deg, rough_m, Qliq, Fw, class_pvt,
                                    calc_dtdl)
        self.assertAlmostEqual(0.088591048678237, dp_dl, places=8)

    def test_3calc_grad_pvt_class(self):
        #    elif oilfield_ == 'Вынгаяхинское':
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
        class_pvt.muob_cP = muob_cP  # Вязкость нефти, из PVT
        class_pvt.qgas_free_sm3day = qgas_free_sm3day
        class_pvt.ksep_fr = ksep_fr
        class_pvt.bwSC_m3m3 = bwSC_m3m3
        class_pvt.salinity_ppm = salinity_ppm

        l_m = 0
        p_atma = 1.536
        t_C = 101.5
        calc_dtdl = False
        pcas_atma = 0.95
        theta_sign = 0
        d_m = 0.079
        theta_deg = 89.95
        rough_m = 0.0001
        Hv_m = 0
        angle_hmes_deg = 89.9500000000073

        Qliq = 62.4
        Fw = 94.4
        pCalc = calc_types.paramCalc(aCorrelation=class_pvt.PVTCorr, aCalcAlongCoord=False, aFlowAlongCoord=False)

        dp_dl = calc_func.calc_grad_pvtcls(l_m, p_atma, t_C, pCalc, d_m, theta_deg, angle_hmes_deg, rough_m, Qliq, Fw, class_pvt, calc_dtdl)
        self.assertEqual(0.0920134015588095, dp_dl)
# 0.09201340155880955 != 0.0920134015588095

class mod_separation_pipe(unittest.TestCase):
    def test_mod_separation_pipe(self):
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
        class_pvt = standing.calc_standing()
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
        pressure = calc_func.mod_separation_pipe(Qliq_, Fw_, Length_, Pcalc_, Calc_along_flow_, class_pvt, Theta_deg, Dtub_,
                                          Hydr_corr_, Tcalc_, Tother_)
        self.assertEqual(2.46295286730971, pressure[0])

    def test_mod_separation_pipe_2(self):
        # Вынгаяхинское месторождение
        gamma_oil = 0.777
        gamma_wat = 1.054
        gamma_gas = 0.749
        rsb_m3m3 = 267
        rp_m3m3 = 7.072
        pb_atma = 243
        pksep_atma = 34.667
        tres_C = 101.5
        bob_m3m3 = 1.582
        muob_cP = 5.58  # Вязкость нефти, из PVT
        bwSC_m3m3 = gamma_wat / 1
        PVTCorr = 0
        ksep_fr = 0.7
        qgas_free_sm3day = 0
        tksep_C = 90
        salinity_ppm = ccalc_pvt.unf_calc_Sal_BwSC_ppm(bwSC_m3m3)
        class_pvt = standing.calc_standing()
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
        Length_ = 9
        Pcalc_ = 9.06033767803739
        Calc_along_flow_ = 0
        Theta_deg = 89.95
        Dtub_ = 79
        Hydr_corr_ = 0
        Tcalc_ = 101.5
        Tother_ = 101.5
        pressure = calc_func.mod_separation_pipe(Qliq_, Fw_, Length_, Pcalc_, Calc_along_flow_, class_pvt, Theta_deg, Dtub_,
                                          Hydr_corr_, Tcalc_, Tother_)
        self.assertEqual(9.91282448991482, pressure[0])
# 9.912849296328108 != 9.91282448991482
#
# Expected :9.91282448991482
# Actual   :9.912849296328108
class q_mix_test_case(unittest.TestCase):

    def test_1st_mf_q_mix(self):
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
        class_pvt = standing.calc_standing()
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

        res = calc_func.q_mix_rc_m3day(Pintake_, Tintake_, Qliq_, Fw_ / 100, class_pvt, calc_func.z_class_select,
                                            calc_func.pseudo_class_select)

        self.assertAlmostEqual(64.28618258473425, class_pvt.q_mix_rc_m3day, places=5)  # 64.28618258473425 != 64.28618258473426

    def test_2nd_q_mix(self):
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
        class_pvt = standing.calc_standing()
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
        calc_func.q_mix_rc_m3day(Pintake_, Tintake_, Qliq_, Fw_ / 100, class_pvt, calc_func.z_class_select,
                                       calc_func.pseudo_class_select)

        self.assertAlmostEqual(66.17282973608943, class_pvt.q_mix_rc_m3day, places=5)  #  66.17282659245636 != 66.17282973608943