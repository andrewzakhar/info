from CPVT.calc_pvt import calc_pvt
import const

class calc_straight_line(calc_pvt):
    # когда PVT_CORRELATION = 2 (данный вариант расчета нефти имеется в юнифолк vba)
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

        ##Debug mode.Linear Rs and bo vs P, p_rb should be specified.
        ##gas properties
        ##ideal gas
        self.Zfactor = 0.95
        self.bg_m3m3 = self.gas_fvf(t, oil_pressure_in_MPa_, self.Zfactor)
        self.mu_gas_cP = 0.0000000001
        ##Set to default.b_rb should be specified by user!
        p_fact = 1
        p_offs = 0
        bubble_point_standing_ = p_rbcalc
        if (oil_pressure_in_MPa_ > (bubble_point_standing_)):
            ##undersaturated oil
            self.rs_m3m3 = self.rsb_m3m3
        else:
            ##saturate
            self.rs_m3m3 = oil_pressure_in_MPa_ / p_rbcalc * self.rsb_m3m3
            ##r_si = GOR_Standing(p_mpa * p_fact, gamma_g, t, gamma_o)

        ##if b_o is not specified by the user then
        ##set b_o so, that oil density, recalculated with r_s would be equal to dead oil density
        if (self.bob_m3m3 < 0):
            if (self.gamma_oil == 0):
                self.bob_m3m3 = None
            else:
                self.bo_m3m3 = (1 + self.rs_m3m3 * (self.gamma_gas * rhoAir) / (self.gamma_oil * rhoRef))
        else:
            if (oil_pressure_in_MPa_ > bubble_point_standing_):
                self.bo_m3m3 = self.bob_m3m3  ##undersaturated oil
            else:
                self.bo_m3m3 = 1.0 + (self.bob_m3m3 - 1.0) * (oil_pressure_in_MPa_ - cPMpa) / (bubble_point_standing_ - cPMpa)
        self.mu_oil_cP = 1

        self.q_oil_rc_m3day, self.q_wat_rc_m3day, self.q_gas_rc_m3day = self.Q_fluid_calc(q_liq, water_сut_share, self.bo_m3m3, self.bw_m3m3,self.rp_m3m3,
                                                                                          self.qgas_free_sm3day, self.rs_m3m3, self.bg_m3m3)


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

        self.ST_oilgas_dyncm, self.ST_watgas_dyncm, self.ST_liqgas_dyncm = self.calc_ST(pksep_atma, tIn_C, water_сut_share, self.gamma_oil)
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