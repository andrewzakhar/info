from CPVT.calc_pvt import calc_pvt
import const
import math_ext as m

class calc_cain(calc_pvt):
    # когда PVT_CORRELATION = 1 - класс, реализующий расчет свойств нефти с использованием
    # набора корреляций на основе McCain
    def calc(self, pksep_atma, tIn_C, q_liq, water_сut_share, z_factor_func, pseudo_crit_func):

        super().calc(pksep_atma, q_liq, tIn_C, water_сut_share, z_factor_func, pseudo_crit_func)

        self.rs_m3m3 = 0
        self.bo_m3m3 = 0
        self.mu_oil_cP = 0
        self.bw_m3m3 = 0
        self.mu_wat_cP = 0
        self.bg_m3m3 = 0
        self.mu_gas_cP = 0
        self.Zfactor = 0
        self.q_oil_rc_m3day = 0
        self.q_wat_rc_m3day = 0
        self.q_gas_rc_m3day = 0

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
        p_fact = 0
        p_offs = 0
        b_fact = 0
        rho_o_sat = 0
        p_rbcalc = 0
        oil_pressure_in_MPa_ = 0

        mcalibr_cP = self.muob_cP
        t = tIn_C + const_t_K_min
        t_res_K = self.tres_C + const_t_K_min
        pseudo_temperature_, pseudo_pressure_ = pseudo_crit_func.press_temp(self.gamma_gas)
        oil_pressure_in_MPa_ = pksep_atma * cPMpa
        ##convert user specifi1ed bubblepoint pb_atma - pressure давление насыщения  (калибровочное значение)
        # 14.084175
        p_rbcalc = self.pb_atma * cPMpa
        ##for saturated oil calibration is applied by application of factor p_fact to input pressure
        ##for undersaturated - by shifting according to p_offs
        ##calculate PVT properties
        ##calculate water properties at current pressure and temperature
        self.bw_m3m3 = self.water_fvf(oil_pressure_in_MPa_, t)

        self.mu_wat_cP = self.water_viscosity(oil_pressure_in_MPa_, t, self.salinity_ppm)

        ##Not Yukos standard set of correlations
        ##Gas properties
        pPr = oil_pressure_in_MPa_ / pseudo_pressure_
        tPr = t / pseudo_temperature_
        args_dic = {'t_pseudo': tPr, 'p_pseudo': pPr}
        self.Zfactor = z_factor_func.z_factor(args_dic)
        self.bg_m3m3 = self.gas_fvf(t, oil_pressure_in_MPa_, self.Zfactor)
        self.mu_gas_cP = self.g_visc(t, oil_pressure_in_MPa_, self.Zfactor, self.gamma_gas)
        ##dead oil viscosity

        dead_oil_viscosity_ = self.dead_oil_viscosity_standing(t, self.gamma_oil)
        ##saturated oil viscosity Beggs Robinson
        saturated_oil_viscosity_beggs_robinson_ = self.saturated_oil_viscosity_beggs_r(self.rsb_m3m3, dead_oil_viscosity_)
        bubble_point_standing_ = self.bubble_point_valko_mccainsi(self.rsb_m3m3, self.gamma_gas, t_res_K, self.gamma_oil)
        ##Standing - debug, remove xxx
        ##p_bi = Bubblepoint_Standing(r_sb, gamma_g, t, gamma_o)
        ##Calculate bubble point correction factor
        if (p_rbcalc > 0):
            ##user specifie
            p_fact = bubble_point_standing_ / p_rbcalc
        else:
            ##not specified, use from correlations
            p_fact = 1

        oil_pressure_in_MPa_ = oil_pressure_in_MPa_ *p_fact
        compressibility_at_bubble_point_pressure_ = self.compressibility_oil_vb(self.rsb_m3m3, self.gamma_gas, t_res_K, self.gamma_oil, oil_pressure_in_MPa_)

        ##Calculate oil formation volume factor correction factor
        if (self.bob_m3m3 > 0):
            ##user specified
            rho_o_sat = self.density_mccainsi(bubble_point_standing_, self.gamma_gas, t_res_K, self.gamma_oil, self.rsb_m3m3,
                                              bubble_point_standing_, compressibility_at_bubble_point_pressure_)
            b_o_sat = self.fvf_mccainsi(self.rsb_m3m3, self.gamma_gas, self.gamma_oil * rhoRef, rho_o_sat)
            b_fact = (self.bob_m3m3 - 1) / (b_o_sat - 1)
        else:
            ##not specified, use from correlations
            b_fact = 1

        bubble_point_standing_ = self.bubble_point_valko_mccainsi(self.rsb_m3m3, self.gamma_gas, t, self.gamma_oil)
        self.rs_m3m3 = self.gor_velardesi(oil_pressure_in_MPa_, bubble_point_standing_, self.gamma_gas, t, self.gamma_oil, self.rsb_m3m3)

        if (oil_pressure_in_MPa_ > bubble_point_standing_):
            ##undersaturated oil
            ##Debug, remove xxx
            ##r_si = r_sb
            ##apply correction to undersaturated oil
            compressibility_at_bubble_point_pressure_ = self.compressibility_oil_vb(self.rsb_m3m3, self.gamma_gas, t, self.gamma_oil, oil_pressure_in_MPa_)
            self.rho_o_sat = self.density_mccainsi(bubble_point_standing_, self.gamma_gas, t, self.gamma_oil, self.rsb_m3m3,
                                              bubble_point_standing_, compressibility_at_bubble_point_pressure_)
            b_o_sat = self.fvf_mccainsi(self.rsb_m3m3, self.gamma_gas, self.gamma_oil * rhoRef, rho_o_sat)
            b_o_sat = b_fact * (self.b_o_sat - 1) + 1
            ##it is assumed that at pressure 1 atm bo = 1
            self.bo_m3m3 = b_o_sat * m.exp_(self.compressibility_at_bubble_point_pressure_ * (self.bubble_point_standing_ - self.oil_pressure_in_MPa_))

        else:
            ##Debug, remove xxx
            ##r_si = GOR_Standing(p_mpa, gamma_g, t, gamma_o)
            ##apply correction to saturated oil
            rho_o = self.density_mccainsi(oil_pressure_in_MPa_, self.gamma_gas, t, self.gamma_oil, self.rs_m3m3,
                                          bubble_point_standing_, compressibility_at_bubble_point_pressure_)
            self.bo_m3m3 = b_fact * (self.fvf_mccainsi(self.rs_m3m3, self.gamma_gas, self.gamma_oil * rhoRef, rho_o) - 1) + 1
            ##it is assumed that at pressure 1 atm bo = 1

        vsc_oil = self.oil_viscosity_standing(self.rsb_m3m3, dead_oil_viscosity_, bubble_point_standing_, bubble_point_standing_)
        if mcalibr_cP > 0:
            if self.rsb_m3m3 < 350:
                self.mu_fact = mcalibr_cP / vsc_oil
            else:
                self.mu_fact = mcalibr_cP / saturated_oil_viscosity_beggs_robinson_
        else:
            self.mu_fact = 1

        if (self.rsb_m3m3 < 350):
            ##Calculate oil viscosity acoording to Standing
            self.mu_oil_cP = self.mu_fact * self.oil_viscosity_standing(self.rs_m3m3, dead_oil_viscosity_, oil_pressure_in_MPa_, bubble_point_standing_)
        else:
            ##Calculate according to BegsRobinson(saturated)and VasquezBegs(undersaturated)
            if (oil_pressure_in_MPa_ > bubble_point_standing_):
                ##undersaturated oil
                self.mu_oil_cP = self.mu_fact * self.oil_viscosity_vasquez_beggs(saturated_oil_viscosity_beggs_robinson_, oil_pressure_in_MPa_, bubble_point_standing_)
            else:
                ##saturated oil
                ##Beggs Robinson
                self.mu_oil_cP = self.mu_fact * self.saturated_oil_viscosity_beggs_r(self.rs_m3m3, dead_oil_viscosity_)

        self.q_oil_rc_m3day, self.q_wat_rc_m3day, self.q_gas_rc_m3day = self.Q_fluid_calc(q_liq, water_сut_share, self.bo_m3m3, self.bw_m3m3,
                                                                                self.rp_m3m3, self.qgas_free_sm3day, self.rs_m3m3, self.bg_m3m3)

        if self.q_oil_rc_m3day == None: self.q_oil_rc_m3day = 0
        if self.q_wat_rc_m3day == None: self.q_wat_rc_m3day = 0
        if self.q_gas_rc_m3day == None: self.q_gas_rc_m3day = 0

        self.q_mix_rc_m3day = self.q_oil_rc_m3day + self.q_wat_rc_m3day + self.q_gas_rc_m3day
        self.qliq_rc_m3day = self.q_wat_rc_m3day + self.q_oil_rc_m3day
        self.fw_rc_fr = 0
        if self.qliq_rc_m3day > 0:
            self.fw_rc_fr = self.q_wat_rc_m3day / self.qliq_rc_m3day
        else:
            self.fw_rc_fr = water_сut_share
        self.mu_liq_cP = self.mu_oil_cP * (1 - self.fw_rc_fr) + self.mu_wat_cP * self.fw_rc_fr

        self.ST_oilgas_dyncm, self.ST_watgas_dyncm, self.ST_liqgas_dyncm = self.calc_ST(pksep_atma, tIn_C, water_сut_share, self.gamma_oil) # p_atma, t_C, water_сut_share, gamma_oil
        self.sigma_liq_Nm = self.ST_liqgas_dyncm * 0.001

        self.rho_oil_rc_kgm3 = 1000 * (self.gamma_oil + self.rs_m3m3 * self.gamma_gas * const.const_rho_air / 1000) / self.bo_m3m3
        self.rho_wat_rc_kgm3 = 1000 * self.gamma_wat / self.bw_m3m3
        self.rho_liq_rc_kgm3 = (1 - water_сut_share) * self.rho_oil_rc_kgm3 + water_сut_share * self.rho_wat_rc_kgm3

        f_g = 0
        self.rho_gas_rc_kgm3 = 0
        if self.q_mix_rc_m3day > 0:
            f_g = self.q_gas_rc_m3day / self.q_mix_rc_m3day
        if self.bg_m3m3 is not None and self.bg_m3m3 > 0:
            self.rho_gas_rc_kgm3 = self.gamma_gas * const.const_rho_air / self.bg_m3m3

        self.rho_mix_rc_kgm3 = self.rho_liq_rc_kgm3 * (1 - f_g) + self.rho_gas_rc_kgm3 * f_g

        self.pbcalc_atma = bubble_point_standing_ / cPMpa / p_fact

        return 0