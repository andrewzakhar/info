from calc_well_param import cesp
import unittest

esp_polynom_head_obj = cesp.esp_polynom([6.67938025106726, -0.03716657517842, 1.08963070530704E-03, -1.53011397338086E-05,
2.78820989024014E-08, 1.28503951587068E-10])

esp_polynom_efficency_obj = cesp.esp_polynom([4.1990495766025E-03, 5.99377724541058E-03, 1.6897242340386E-05, -2.73132096997276E-07,
 9.14180916880957E-11, 1.24234461289717E-12])

esp_polynom_power_obj = cesp.esp_polynom([6.29041697452997E-02, -7.19345665833556E-04, 2.87765660920731E-05, -3.07744767249128E-07,
1.38139467562467E-09, -1.94913074195634E-12])

class get_ESP_head_m_test_case(unittest.TestCase):
    def test_get_ESP_head_m(self):
        new_esp = cesp.esp(id_pump = 743, manufacturer_name='Новомет', pump_name='mypump', freq_hz=50, esp_nom_rate_m3day=59, esp_max_rate_m3day=124,
                           esp_polynom_head_obj=esp_polynom_head_obj, esp_polynom_efficency_obj=esp_polynom_efficency_obj,
                           esp_polynom_power_obj=esp_polynom_power_obj)
        new_esp.freq_hz = 55.1
        new_esp.freq_hz = 55.1
        head_val = new_esp.get_esp_head_m(aqliq_m3day=65.018)
        self.assertAlmostEqual(6.76028599134472, head_val,  places=13)

class calc_corrVisc_petrInst_test_case(unittest.TestCase):
    def test_calc_corrVisc_petrInst(self):
        new_esp = cesp.esp(id_pump = 743, manufacturer_name='Новомет', pump_name='mypump', freq_hz=50, esp_nom_rate_m3day=59, esp_max_rate_m3day=124,
                           esp_polynom_head_obj=esp_polynom_head_obj, esp_polynom_efficency_obj=esp_polynom_efficency_obj,
                           esp_polynom_power_obj=esp_polynom_power_obj)
        new_esp.mu_cSt = 7.76
        new_esp.correct_visc = True
        new_esp.freq_hz = 55.1
        new_esp.calc_corrVisc_petrInst(aq_mix = 65.3192193823106)
        self.assertAlmostEqual(136.648, new_esp.max_rate_m3day,  places=10)
        self.assertAlmostEqual(0.981367394278045, new_esp.corr_visc_h,  places=3)
        self.assertAlmostEqual(0.916369927579911, new_esp.corr_visc_eff,  places=3)
        self.assertAlmostEqual(1.09126234930139, new_esp.corr_visc_pow,  places=3)
        self.assertAlmostEqual(0.988413991883029, new_esp.corr_visc_q,  places=3)

class esp_power_w_test_case(unittest.TestCase):
    def test_calc_esp_power_w(self):
        new_esp = cesp.esp(id_pump=743, manufacturer_name='Новомет', pump_name='mypump', freq_hz=50,
                           esp_nom_rate_m3day=59, esp_max_rate_m3day=124,
                           esp_polynom_head_obj=esp_polynom_head_obj, esp_polynom_efficency_obj=esp_polynom_efficency_obj,
                           esp_polynom_power_obj=esp_polynom_power_obj)
        new_esp.mu_cSt = 7.76
        new_esp.correct_visc = True
        new_esp.freq_hz = 55.1
        new_esp.stage_num = 296
        power_w = new_esp.esp_power_w(aqliq_m3day=65.3192193823106)
        self.assertAlmostEqual(31659.4526789701, power_w,  places=10)

    def test_polinom_solver(self):
        qliq = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 66]
        head = [4.88, 4.73, 4.66, 4.61, 4.52, 4.35, 4.10, 3.74, 3.28, 2.73, 2.11, 1.45, 0.77, 0]
        _esp_polynom_head_obj = cesp.esp_polynom([4.87960179118, -0.04329052576, 0.00328204652,
                                                  -0.00012527705, 0.00000127703, -0.00000000362])

        a = cesp.polinom_solver(qliq, head, 5)
        self.assertAlmostEqual(_esp_polynom_head_obj.coef0, a[0],  places=8)
        self.assertAlmostEqual(_esp_polynom_head_obj.coef1, a[1],  places=8)
        self.assertAlmostEqual(_esp_polynom_head_obj.coef2, a[2],  places=8)
        self.assertAlmostEqual(_esp_polynom_head_obj.coef3, a[3],  places=8)
        self.assertAlmostEqual(_esp_polynom_head_obj.coef4, a[4],  places=8)
        self.assertAlmostEqual(_esp_polynom_head_obj.coef5, a[5],  places=8)

esp_polynom_hd_obj = cesp.esp_polynom([6.67938025106726, -0.03716657517842, 1.08963070530704E-03, -1.53011397338086E-05,
                                         2.78820989024014E-08, 1.28503951587068E-10])

esp_polynom_eff_obj = cesp.esp_polynom([1.25187208057161E-03, 1.25417590382097E-02, 8.35978586124621E-06, -1.53042920205554E-06,
                                              5.18177482397258E-09, 3.73609276785907E-13])

esp_polynom_pow_obj = cesp.esp_polynom([6.29041697452997E-02, -7.19345665833556E-04, 2.87765660920731E-05, -3.07744767249128E-07,
                                          1.38139467562467E-09, -1.94913074195634E-12])

class esp_efficiency_rp_test_case (unittest.TestCase):
    def test_esp_efficiency_rp(self):


        new_esp = cesp.esp(id_pump=743, manufacturer_name='Новомет', pump_name='ВНН5-59', freq_hz=50,
                           esp_nom_rate_m3day=59, esp_max_rate_m3day=124,
                           esp_polynom_head_obj=esp_polynom_hd_obj,
                           esp_polynom_efficency_obj=esp_polynom_eff_obj,
                           esp_polynom_power_obj=esp_polynom_pow_obj)

        new_esp.mu_cSt = 5.58
        new_esp.correct_visc = True
        new_esp.freq_hz = 55.1
        ef_rp = new_esp.esp_efficiency_rp(aqliq_m3day=66.0810412253359)
        self.assertAlmostEqual(0.4845668570829677, ef_rp, places=10)

