import calc_func
from xalglib import xalglib
import math
import calc_types
from CPVT.standing import calc_standing
import ccalc_pvt

x_gr = 0
y_gr = 1
class CPipe():


    def pipe_atma_pvtcls(self,a_qliq_sm3day, a_fw_perca_, a_length_m, a_pcalc_atma, class_pvt,
                         a_theta_deg, a_d_mm, a_t_calc_C):


        a_roughness_m = 0.00001
        # расчет распределения давления и температуры в трубе с использованием многофазных корреляций
        # qliq_sm3day - дебит жидкости в поверхностных условиях
        # fw_perc     - обводненность
        # Length_m    - Длина трубы, измеренная, м
        # calc_along_flow - флаг направления расчета относительно потока
        #     если = 1 то расчет по потоку
        #     если = 0 то расчет против потока
        # Pcalc_atma  - давление с которого начинается расчет, атм
        #               граничное значение для проведения расчета
        # aPVTdic - словарь с параметрами PVT
        # theta_deg   - угол направления потока к горизонтали
        #               (90 - вертикальная труба поток вверх
        #                -90 - вертикальная труба поток вниз)
        #               может принимать отрицательные значения
        # d_mm        - внутренний диаметр трубы
        # hydr_corr    - гидравлическая корреляция, H_CORRELATION
        #                  BeggsBrill = 0
        #                  Ansari = 1
        #                  Unified = 2
        #                  Gray = 3
        #                  HagedornBrown = 4
        #                  SakharovMokhov = 5
        # t_calc_C     - температура в точке где задано давление, С
        # Tother_C    - температура на другом конце трубы
        #               по умолчанию температура вдоль трубы постоянна
        #               если задано то меняется линейно по трубе
        # roughness_m - шероховатость трубы
        length_m = None
        if a_length_m != a_length_m or a_length_m == None or a_length_m == 0:
            return None
        else:
            length_m = abs(a_length_m)
        # habs_curve_m - инклинометрия - зависимость значений вертикальной глубины от измеренной
        # Здесь вертикальная отметка
        habs_curve_m = length_m * abs(math.sin(a_theta_deg / 180 * math.pi))

        # diam_data_mm - значения диаметров от измеренной глубины
        #             первый столбец - измеренная глубина, м
        #             второй столбец - диаметр трубы, мм - применяется от текущего значения глубины и до следующего
        #             если передано одно число - то будет задан постоянный диаметр
        diam_curve_mm = []
        diam_curve_mm.append([0, a_d_mm, a_roughness_m])
        diam_curve_mm.append([length_m, a_d_mm, a_roughness_m])

        # Расчет угла на текущей глубине
        sina = habs_curve_m / length_m
        cosa = math.sqrt(max(1 - sina ** 2, 0))
        if cosa == 0:
            ang = 90 * sina
        else:
            ang = math.atan(sina / cosa) * 180 / math.pi
        angle_init_deg = ang

        # init params
        pCalc = calc_types.paramCalc(aCorrelation= class_pvt.PVTCorr, aCalcAlongCoord=False,
                                     aFlowAlongCoord=False)
        calc_dtdl = False
        rough_m = 0.0001
        # Преобразуем в метры
        d_m = a_d_mm / 1000

        y = [a_pcalc_atma]
        x = [0, length_m]
        eps = 0.1
        h = length_m

        def ode_function_1_diff(y, x, dy, param):
            #
            # this callback calculates f(y[],x)=-y[0]
            #
            dp_dl = calc_func.calc_grad_pvtcls(x, y[0], a_t_calc_C, pCalc, d_m, a_theta_deg, angle_init_deg,
                                               rough_m,
                                               a_qliq_sm3day, a_fw_perca_, class_pvt, calc_dtdl)
            dy[0] = dp_dl
            # print(f'y = {y}, x = {x}, dy[0] = {dy[0]}')
            return

        s = xalglib.odesolverrkck(y, x, eps, h)
        xalglib.odesolversolve(s, ode_function_1_diff)
        m, xtbl, ytbl, rep = xalglib.odesolverresults(s)
        pressure = ytbl[m - 1][0]
        return pressure

    def pipe_atma_modern(self, a_qliq_sm3day, a_fw_perca_, a_length_m, a_pcalc_atma, a_calc_along_flow, class_pvt,
                         a_theta_deg, a_d_mm, a_hydr_corr, a_t_calc_C, a_Tother_C):


        # функция модификации свойств нефти после сепарации
        # удаление части газа меняет свойства нефти - причем добавление газа свойства не трогает
        # на входе условия при которых проходила сепарация
        z_class = calc_func.z_factor_2015_kareem()
        pseudo_class = calc_func.pseudo_standing()
        n = 10
        #  промежуточные переменные для сохранения значений этих параметров
        #  эти параметры должны возвращаться с теми значениями, с которыми они пришли
        rp_m3m3_r = class_pvt.rp_m3m3
        rsb_m3m3_r = class_pvt.rsb_m3m3
        bob_m3m3_r = class_pvt.bob_m3m3
        pb_atma_r = class_pvt.pb_atma
        rp_m3m3 = class_pvt.rp_m3m3

        class_pvt.rsb_m3m3 = class_pvt.rp_m3m3  # rsb_m3m3 должен быть равен rp_m3m3 вне зависимости какое изначально он значение имел до этого

        pb_atma = class_pvt.pb_atma
        Ksep = class_pvt.ksep_fr
        Rs = 0
        Bo = 0
        Rpnew_with_Ksep = 0
        Rpnew_Ksep_1 = 0
        pb_atma_tab = 0
        Rpnew = 0
        GasSol = 1  # передается в функцию mod_after_separation как аргумент
        GasGoesIntoSolution = 1  # проинициализирован в u7_types

        Rs, Bo = calc_func.calc_rs_bo_m3m3(class_pvt.pksep_atma, class_pvt.tksep_C,  a_qliq_sm3day, a_fw_perca_, class_pvt, z_class,
                                 pseudo_class)

        Rpnew_with_Ksep = rp_m3m3 - (rp_m3m3 - Rs) * Ksep
        Rpnew_Ksep_1 = rp_m3m3 - (rp_m3m3 - Rs)

        Delta = (pb_atma - 1) / n
        i = 0
        Tintake_tres_C = 101.5
        func_array_1 = [[], []]
        func_array_2 = [[], []]
        while i <= n:
            pb_atma_tab = 1 + Delta * i
            rsb_m3m3_tab, Bo_m3m3_tab = calc_func.calc_rs_bo_m3m3(pb_atma_tab, Tintake_tres_C, a_qliq_sm3day, a_fw_perca_, class_pvt, z_class,
                                 pseudo_class)
            # добавляем элементы в два массива. Они будут представлять из себя два "графика"
            func_array_1[x_gr].append(rsb_m3m3_tab)  # будет представлять себя массив точек X
            func_array_1[y_gr].append(pb_atma_tab)  # будет представлять себя массив точек Y
            func_array_2[x_gr].append(rsb_m3m3_tab)
            func_array_2[y_gr].append(Bo_m3m3_tab)
            i = i + 1

        if GasSol == GasGoesIntoSolution:
            Rpnew = Rpnew_with_Ksep
        else:
            Rpnew = Rpnew_Ksep_1

        if Rpnew < class_pvt.rsb_m3m3:
            class_pvt.pb_atma, class_pvt.bob_m3m3 = calc_func.getPoint(Rpnew, i, func_array_1, func_array_2)
            class_pvt.rsb_m3m3 = Rpnew

        class_pvt.rp_m3m3 = Rpnew_with_Ksep

        pressure = self.pipe_atma_pvtcls(a_qliq_sm3day, a_fw_perca_, a_length_m, a_pcalc_atma, class_pvt, a_theta_deg, a_d_mm, a_t_calc_C)

        # возвращаем в "исходное состояние"
        class_pvt.rp_m3m3 = rp_m3m3_r
        class_pvt.rsb_m3m3 = rsb_m3m3_r
        class_pvt.pb_atma = pb_atma_r
        class_pvt.bob_m3m3 = bob_m3m3_r

        #    PVTdic['rp_m3m3'] = rp_m3m3_r
        #    PVTdic['rsb_m3m3'] = rsb_m3m3_r
        #    PVTdic['pb_atma'] = pb_atma_r
        #    PVTdic['bob_m3m3'] = bob_m3m3_r

        return pressure