import os
import pandas as pd
import math
from xalglib import xalglib
import numpy as np
#import common.myLogger as log
import common.vr_logger as log
from calc_well_param import const


class esp_polynom():
    __coef0 = None

    @property
    def coef0(self):
        return self.__coef0

    __coef1 = None

    @property
    def coef1(self):
        return self.__coef1

    __coef2 = None

    @property
    def coef2(self):
        return self.__coef2

    __coef3 = None

    @property
    def coef3(self):
        return self.__coef3

    __coef4 = None

    @property
    def coef4(self):
        return self.__coef4

    __coef5 = None

    @property
    def coef5(self):
        return self.__coef5

    def __init__(self, a):
        if len(a) == 6:
            self.__coef0 = a[0]
            self.__coef1 = a[1]
            self.__coef2 = a[2]
            self.__coef3 = a[3]
            self.__coef4 = a[4]
            self.__coef5 = a[5]
        self.result = None

    def calc(self, arg):
        self.result = self.__coef0 + \
                      self.__coef1 * arg + \
                      self.__coef2 * arg ** 2 + \
                      self.__coef3 * arg ** 3 + \
                      self.__coef4 * arg ** 4 + \
                      self.__coef5 * arg ** 5
        return self.result


class esp:
    '''
     класс для моделирования работы погружной части ЭЦН
     описывает работу набора одинаковых ступеней
    '''

    __id_pump = None
    '''ID  из базы'''

    @property
    def id_pump(self):
        return self.__id_pump

    __manufacturer_name = None
    '''производитель насоса (справочный параметр)'''

    @property
    def manufacturer_name(self):
        return self.__manufacturer_name

    __pump_name = None
    '''название насоса (справочный параметр)'''

    @property
    def pump_name(self):
        return self.__pump_name

    __esp_freq_hz = None
    '''частота насоса для номинальной характеристики в базе'''

    @property
    def esp_freq_hz(self):
        return self.__esp_freq_hz

    __freq_hz = None
    '''частота насоса для расчета'''

    @property
    def freq_hz(self):
        return self.__freq_hz

    @freq_hz.setter
    def freq_hz(self, freq_hz):
        self.__freq_hz = freq_hz
        if self.esp_nom_rate_m3day != None and self.esp_freq_hz != None and self.esp_freq_hz != 0 and freq_hz != None:
            self.__nom_rate_m3day = self.esp_nom_rate_m3day * self.__freq_hz / self.esp_freq_hz
            if self.esp_max_rate_m3day != None:
                self.__max_rate_m3day = self.esp_max_rate_m3day * self.__freq_hz / self.esp_freq_hz
        else:
            self.__nom_rate_m3day = None
            self.__max_rate_m3day = None

    __esp_nom_rate_m3day = None
    '''номинальный дебит насоса (из базы)'''

    @property
    def esp_nom_rate_m3day(self):
        return self.__esp_nom_rate_m3day

    __nom_rate_m3day = None
    '''номинальный дебит насоса расчетный от частоты'''

    @property
    def nom_rate_m3day(self):
        return self.__nom_rate_m3day

    __esp_max_rate_m3day = None
    '''максимальный дебит насоса расчетный от частоты'''

    @property
    def esp_max_rate_m3day(self):
        return self.__esp_max_rate_m3day

    __stage_num = 1
    '''Количество ступеней'''

    @property
    def stage_num(self):
        return self.__stage_num

    @stage_num.setter
    def stage_num(self, stage_num):
        self.__stage_num = stage_num

    __max_stages_number = None
    '''максимальное количество ступеней в насосе (из базы)'''

    @property
    def max_stages_number(self):
        return self.__max_stages_number

    @max_stages_number.setter
    def max_stages_number(self, max_stages_number):
        self.__max_stages_number = max_stages_number

    __head = None
    ''' напор '''

    @property
    def head(self):
        return self.__head

    @head.setter
    def head(self, head):
        self.__head = head

    __optimum_min_rate_m3day = None
    '''границы оптимального диапазона для насоса - минимум'''

    @property
    def optimum_min_rate_m3day(self):
        return self.__optimum_min_rate_m3day

    @optimum_min_rate_m3day.setter
    def optimum_min_rate_m3day(self, optimum_min_rate_m3day):
        self.__optimum_min_rate_m3day = optimum_min_rate_m3day

    __optimum_max_rate_m3day = None
    '''границы оптимального диапазона  - максимум'''

    @property
    def optimum_max_rate_m3day(self):
        return self.__optimum_max_rate_m3day

    @optimum_max_rate_m3day.setter
    def optimum_max_rate_m3day(self, optimum_max_rate_m3day):
        self.__optimum_max_rate_m3day = optimum_max_rate_m3day

    __max_rate_m3day = None
    '''максимальный дебит насоса расчетный '''

    @property
    def max_rate_m3day(self):
        return self.__max_rate_m3day

    __correct_visc = False
    '''
    учет вязкости
    При создании экземпляра вязкость не может учитываться, т.к. сама вязкость еще не установлена.
    '''

    @property
    def correct_visc(self):
        return self.__correct_visc

    @correct_visc.setter
    def correct_visc(self, correct_visc):
        self.__correct_visc = correct_visc
        if self.__correct_visc != True:
            self.__corr_visc_h = 1
            self.__corr_visc_q = 1
            self.__corr_visc_pow = 1
            self.__corr_visc_eff = 1

    __mu_cSt = None
    '''вязкость смеси'''

    @property
    def mu_cSt(self):
        return self.__mu_cSt

    @mu_cSt.setter
    def mu_cSt(self, mu_cSt):
        if mu_cSt != mu_cSt or mu_cSt == None or mu_cSt < 5:
            self.correct_visc = False
            self.__mu_cSt = None
        else:
            self.correct_visc = True
            self.__mu_cSt = mu_cSt

    __corr_visc_h = 1
    ''' поправочный коэффициент для напорной характеристики на вязкость для текущего дебита и текущего расчета '''

    @property
    def corr_visc_h(self):
        return self.__corr_visc_h

    @corr_visc_h.setter
    def corr_visc_h(self, corr_visc_h):
        if corr_visc_h != corr_visc_h or not corr_visc_h or corr_visc_h < 0:
            self.__corr_visc_h = None
        else:
            self.__corr_visc_h = corr_visc_h

    __corr_visc_q = 1
    ''' поправочный коэффициент для дебита '''

    @property
    def corr_visc_q(self):
        return self.__corr_visc_q

    @corr_visc_q.setter
    def corr_visc_q(self, corr_visc_q):
        self.__corr_visc_q = corr_visc_q

    __corr_visc_pow = 1
    ''' поправочный коэффициент для мощности '''

    @property
    def corr_visc_pow(self):
        return self.__corr_visc_pow

    @corr_visc_pow.setter
    def corr_visc_pow(self, corr_visc_pow):
        self.__corr_visc_pow = corr_visc_pow

    __corr_visc_eff = 1

    @property
    def corr_visc_eff(self):
        return self.__corr_visc_eff

    @corr_visc_eff.setter
    def corr_visc_eff(self, corr_visc_eff):
        self.__corr_visc_eff = corr_visc_eff

    __pump_depth_without_nkt = None

    @property
    def pump_depth_without_nkt(self):
        return self.__pump_depth_without_nkt

    @pump_depth_without_nkt.setter
    def pump_depth_without_nkt(self, pump_depth_without_nkt):
        self.__pump_depth_without_nkt = pump_depth_without_nkt

    __pump_depth_m = None

    @property
    def pump_depth_m(self):
        return self.__pump_depth_m

    @pump_depth_m.setter
    def pump_depth_m(self, pump_depth_m):
        self.__pump_depth_m = pump_depth_m

    __esp_polynom_head_obj = None

    @property
    def esp_polynom_head_obj(self):
        return self.__esp_polynom_head_obj

    @esp_polynom_head_obj.setter
    def esp_polynom_head_obj(self, esp_polynom_head_obj):
        self.__esp_polynom_head_obj = esp_polynom_head_obj

    __esp_polynom_efficency_obj = None

    @property
    def esp_polynom_efficency_obj(self):
        return self.__esp_polynom_efficency_obj

    @esp_polynom_efficency_obj.setter
    def esp_polynom_efficency_obj(self, esp_polynom_efficency_obj):
        self.__esp_polynom_efficency_obj = esp_polynom_efficency_obj

    __esp_polynom_power_obj = None

    @property
    def esp_polynom_power_obj(self):
        return self.__esp_polynom_power_obj

    @esp_polynom_power_obj.setter
    def esp_polynom_power_obj(self, esp_polynom_power_obj):
        self.__esp_polynom_power_obj = esp_polynom_power_obj

    def __init__(self, id_pump, manufacturer_name, pump_name, freq_hz, esp_nom_rate_m3day, esp_max_rate_m3day,
                 esp_polynom_head_obj, esp_polynom_efficency_obj, esp_polynom_power_obj):
        try:
            self.__id_pump = id_pump
            self.__manufacturer_name = manufacturer_name
            self.__pump_name = pump_name
            if freq_hz is None:
                self.__esp_freq_hz = 50
            else:
                self.__esp_freq_hz = freq_hz
            self.__esp_nom_rate_m3day = esp_nom_rate_m3day
            self.__esp_max_rate_m3day = esp_max_rate_m3day
            self.esp_polynom_head_obj = esp_polynom_head_obj
            self.esp_polynom_efficency_obj = esp_polynom_efficency_obj
            self.esp_polynom_power_obj = esp_polynom_power_obj
            log.logger.debug(
             f'Module: "{__name__}"       Function: "{self.__init__.__name__}"       Current parameters: id_pump = {id_pump}, manufacturer_name = {manufacturer_name}, pump_name = {pump_name}, freq_hz = {freq_hz}, esp_nom_rate_m3day = {esp_nom_rate_m3day}, esp_max_rate_m3day = {esp_max_rate_m3day}, esp_polynom_head_obj = {esp_polynom_head_obj}, esp_polynom_efficency_obj = {esp_polynom_efficency_obj}, esp_polynom_power_obj = {esp_polynom_power_obj}')
        except:
           log.logger.exception(f'Module: "{__name__}"       Function: "{self.__init__.__name__}"')
           log.logger.error(
               f'Module: "{__name__}"       Function: "{self.__init__.__name__}"       Current parameters: id_pump = {id_pump}, manufacturer_name = {manufacturer_name}, pump_name = {pump_name}, freq_hz = {freq_hz}, esp_nom_rate_m3day = {esp_nom_rate_m3day}, esp_max_rate_m3day = {esp_max_rate_m3day}, esp_polynom_head_obj = {esp_polynom_head_obj}, esp_polynom_efficency_obj = {esp_polynom_efficency_obj}, esp_polynom_power_obj = {esp_polynom_power_obj}')

    def properties_to_str(self):
        str = f"id_pump = {self.id_pump}, manufacturer_name = {self.manufacturer_name}, pump_name = {self.pump_name}, " \
              f"esp_freq_hz = {self.esp_freq_hz}, freq_hz = {self.freq_hz}, esp_nom_rate_m3day = {self.esp_nom_rate_m3day}, " \
              f"nom_rate_m3day = {self.nom_rate_m3day}, esp_max_rate_m3day = {self.esp_max_rate_m3day}, stage_num = {self.stage_num}, " \
              f"max_stages_number = {self.max_stages_number}, head = {self.head}, optimum_min_rate_m3day = {self.optimum_min_rate_m3day}, " \
              f"optimum_max_rate_m3day = {self.optimum_max_rate_m3day}, max_rate_m3day = {self.max_rate_m3day}, correct_visc = {self.correct_visc}, " \
              f"mu_cSt = {self.mu_cSt}, corr_visc_h = {self.corr_visc_h}, corr_visc_q = {self.corr_visc_q}, corr_visc_pow = {self.corr_visc_pow}, " \
              f"corr_visc_eff = {self.corr_visc_eff}, pump_depth_m = {self.pump_depth_m}, esp_polynom_head_obj = {self.esp_polynom_head_obj}, " \
              f"esp_polynom_efficency_obj = {self.esp_polynom_efficency_obj}, esp_polynom_power_obj = {self.esp_polynom_power_obj}"
        return str

    def calc_corrVisc_petrInst(self, aq_mix):
        try:
            '''
                метод для расчета корректировки напорной характеристики УЭЦН на вязкость для текущего насоса
                расчет для одной ступени
            '''
            if self.correct_visc == False:
                return None
            GAMMA = None
            QwBEP_100gpm = None
            HwBEP_ft = None
            Qstar = None
            Q0 = None
            Q0_6 = None
            Q0_8 = None
            Q1_0 = None
            Q1_2 = None
            qmax = None
            H0 = None
            H0_6 = None
            H0_8 = None
            H1_0 = None
            H1_2 = None
            Hmax = None

            QwBEP_100gpm = self.nom_rate_m3day * const.const_convert_m3day_gpm
            tmp_esp = self.clone()
            tmp_esp.correct_visc = False
            tmp_esp.stage_num = 1
            HwBEP_ft = tmp_esp.get_esp_head_m(self.nom_rate_m3day) * const.const_convert_m_ft
            if HwBEP_ft == 0:
                self.corr_visc_h = 1
                return
            # nu_cSt = visc_cP / fluid.rho_oil_sckgm3
            GAMMA = - 7.5946 + 6.6504 * math.log(HwBEP_ft) + 12.8429 * math.log(QwBEP_100gpm)
            Qstar = math.exp((39.5276 + 26.5606 * math.log(self.mu_cSt) - GAMMA) / 51.6565)
            self.corr_visc_q = 1 - 4.0327 * 10 ** (- 3) * Qstar - 1.724 * 10 ** (- 4) * Qstar ** 2
            if (self.corr_visc_q < 0):
                self.corr_visc_h = None
                return
            self.corr_visc_eff = 1 - 3.3075 * 10 ** (- 2) * Qstar + 2.8875 * 10 ** (- 4) * Qstar ** 2
            self.corr_visc_pow = 1 / self.corr_visc_eff
            Q0 = 0
            Q1_0 = self.nom_rate_m3day * self.corr_visc_q
            H1_0 = 1 - 7.00763 * 10 ** (-3) * Qstar - 1.41 * 10 ** (-5) * Qstar ** 2
            Q0_8 = Q1_0 * 0.8
            H0_8 = 1 - 4.4726 * 10 ** (-3) * Qstar - 4.18 * 10 ** (-5) * Qstar ** 2
            Q0_6 = Q1_0 * 0.6
            H0_6 = 1 - 3.68 * 10 ** (-3) * Qstar - 4.36 * 10 ** (-5) * Qstar ** 2
            Q1_2 = Q1_0 * 1.2
            H1_2 = 1 - 9.01 * 10 ** (-3) * Qstar + 1.31 * 10 ** (-5) * Qstar ** 2
            qmax = self.max_rate_m3day * self.corr_visc_q
            # Hmax = H1_2
            if qmax < Q1_2:
                self.corr_visc_h = None
                return
                # тут что то не так с характеристиков насоса - номинальный и максимальный дебит не соответствуют друг другу
            qd_curve_q = [Q1_2, Q1_0, Q0_8, Q0_6]
            qd_curve_h = [H1_2, H1_0, H0_8, H0_6]
            spline = xalglib.spline1dbuildcubic(qd_curve_q, qd_curve_h)
            H0 = xalglib.spline1dcalc(spline, Q0)
            if H0 < 0:
                H0 = H0_6
            qd_curve_q.append(Q0)
            qd_curve_h.append(H0)
            spline = xalglib.spline1dbuildcubic(qd_curve_q, qd_curve_h)
            self.corr_visc_h = xalglib.spline1dcalc(spline, aq_mix)
            log.logger.debug(
                f'Module: "{__name__}"       Function: "{self.calc_corrVisc_petrInst.__name__}"       Current parameters: aq_mix = {aq_mix}')
        except:
            log.logger.exception(f'Module: "{__name__}"       Function: "{self.calc_corrVisc_petrInst.__name__}"')
            log.logger.error(
                f'Module: "{__name__}"       Function: "{self.calc_corrVisc_petrInst.__name__}"       Current parameters: aq_mix = {aq_mix}')

    def clone(self):
        try:
            """клонировать экземпляр"""
            esp_copy = esp(self.id_pump, self.__manufacturer_name, self.__pump_name, self.__esp_freq_hz,
                           self.__esp_nom_rate_m3day, self.__esp_max_rate_m3day,
                           esp_polynom_head_obj=self.esp_polynom_head_obj,
                           esp_polynom_efficency_obj=self.esp_polynom_efficency_obj,
                           esp_polynom_power_obj=self.esp_polynom_power_obj)
            esp_copy.corr_visc_h = self.corr_visc_h
            esp_copy.corr_visc_q = self.corr_visc_q
            esp_copy.corr_visc_pow = self.corr_visc_pow
            esp_copy.corr_visc_eff = self.corr_visc_eff
            esp_copy.max_stages_number = self.max_stages_number
            esp_copy.optimum_min_rate_m3day = self.optimum_min_rate_m3day
            esp_copy.optimum_max_rate_m3day = self.optimum_max_rate_m3day
            esp_copy.freq_hz = self.freq_hz
            esp_copy.correct_visc = self.correct_visc
            return esp_copy
        except:
            log.logger.exception(f'Module: "{__name__}"       Function: "{self.clone.__name__}"')

    def get_esp_head_m(self, aqliq_m3day):
        try:
            """ номинальный напор ЭЦН (на основе каталога ЭЦН)"""
            """ учитывается поправка на вязкость"""
            if aqliq_m3day != aqliq_m3day or aqliq_m3day == None or aqliq_m3day < 0:
                return None

            b = None
            stage_num = None
            if self.max_rate_m3day != None and aqliq_m3day > self.max_rate_m3day != None \
                    and aqliq_m3day > self.max_rate_m3day * 1.5:
                return None

            if self.correct_visc == True:
                self.calc_corrVisc_petrInst(aqliq_m3day)
            aqliq_m3day = aqliq_m3day / self.corr_visc_q
            b = self.esp_freq_hz / self.freq_hz
            fn_return_value = b ** (- 2) * self.stage_num * self.esp_polynom_head_obj.calc(b * aqliq_m3day)
            if fn_return_value < 0:
                fn_return_value = 0
            fn_return_value = fn_return_value * self.corr_visc_h
            log.logger.debug(
                f'Module: "{__name__}"       Function: "{self.get_esp_head_m.__name__}"       Current parameters: aqliq_m3day = {aqliq_m3day}, fn_return_value = {fn_return_value}')
            return fn_return_value
        except:
            log.logger.exception(f'Module: "{__name__}"       Function: "{self.get_esp_head_m.__name__}"')
            log.logger.error(
                f'Module: "{__name__}"       Function: "{self.get_esp_head_m.__name__}"       Current parameters: aqliq_m3day = {aqliq_m3day}')

    def esp_power_w(self, aqliq_m3day, c_calibr_rate=1, c_calibr_power=1):
        try:
            ''' мощность УЭЦН номинальная потребляемая, учитывается поправка на вязкость
             qliq_m3day - дебит жидкости в условиях насоса (стенд)
             num_stages  - количество ступеней
             freq_Hz       - частота вращения насоса
             pump_id     - номер насоса в базе данных
             mu_cSt     - вязкость жидкости
             c_calibr_rate - поправочный коэффициент (множитель) на подачу насоса.
             c_calibr_power - поправочный коэффициент (множитель) на мощность насоса.
            '''
            nconsumption_w = None
            eff1 = None
            esp_power_w1 = None
            # возвращаемое значение
            pow_w = None
            if aqliq_m3day != aqliq_m3day or aqliq_m3day == None or aqliq_m3day < 0:
                return None
            liq_m3day = aqliq_m3day / c_calibr_rate
            if aqliq_m3day > self.max_rate_m3day:
                # assume that for high rate power consumption will not be less that at max rate
                liq_m3day = self.max_rate_m3day
            if self.correct_visc == True:
                self.calc_corrVisc_petrInst(liq_m3day)
            # делаем коррекцию по вязкости
            liq_m3day = liq_m3day / self.corr_visc_q

            b = self.esp_freq_hz / self.freq_hz
            pow_w = 1000 * b ** (-3) * self.stage_num * self.esp_polynom_power_obj.calc(b * liq_m3day)
            if pow_w < 0:
                pow_w = None
            else:
                pow_w = pow_w * self.corr_visc_pow
                pow_w = pow_w * c_calibr_power

            # поскольку в базе Роспампа выявлены насосы с некорректными характеристиками
            # проведем тут проверку - рассчитаем мощность через КПД и сравним с исходным значением в базе данных
            # nconsumption_w = self.get_ESP_head_m(liq_m3day)
            # if nconsumption_w != None:
            #     nconsumption_w = nconsumption_w * const.const_g * const.const_rho_ref * liq_m3day * const.const_convert_m3day_m3sec
            log.logger.debug(
                f'Module: "{__name__}"       Function: "{self.esp_power_w.__name__}"       Current parameters: aqliq_m3day = {aqliq_m3day}, c_calibr_rate = {c_calibr_rate}, c_calibr_power = {c_calibr_power}, pow_w = {pow_w}')
            return pow_w
        except:
            log.logger.exception(f'Module: "{__name__}"       Function: "{self.esp_power_w.__name__}"')
            log.logger.error(
                f'Module: "{__name__}"       Function: "{self.esp_power_w.__name__}"       Current parameters: aqliq_m3day = {aqliq_m3day}, c_calibr_rate = {c_calibr_rate}, c_calibr_power = {c_calibr_power}')

    def esp_efficiency_rp (self, aqliq_m3day):

        if aqliq_m3day != aqliq_m3day or aqliq_m3day == None or aqliq_m3day < 0:
            return 0

        if aqliq_m3day > self.max_rate_m3day:
            return 0

        if self.correct_visc == True:
            self.calc_corrVisc_petrInst(aqliq_m3day)

        aqliq_m3day  = aqliq_m3day/ self.corr_visc_q

        b = self.esp_freq_hz / self.freq_hz
        ef_rp = self.esp_polynom_efficency_obj.calc(b * aqliq_m3day)
        if ef_rp < 0:
            ef_rp = 0
        else:
            ef_rp = ef_rp * self.__corr_visc_eff
        return ef_rp



def polinom_solver(x, y, n):
    ''' Функция рассчитывает коэффициенты полинома для аппроксимации функции вида y = a0 + a1*x + a2*x^2 + ... an*x^n
    x, y - известная зависимость y от x - одномерные массивы
    n - максимальная степень полинома
    '''
    if not x or not y or len(x) == 0 or len(y) == 0:
        return None
    a = np.polyfit(x, y, n)
    a = a[::-1]  # Развернем массив, т.к функция polyfit возвращает
    # под индексом 0 коэффициент 5-й степени
    return a
