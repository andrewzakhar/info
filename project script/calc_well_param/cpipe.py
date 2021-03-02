import os
import sys
from xalglib import xalglib
import math

import math_ext as ma
import const
import calc_types
import calc_func

sys.path.append(os.path.abspath(''))

def transform(aTrajectory, multY=1, sumY=0, multX=1, sumX=0):
    # преобразует кривую с использованием линейного преобразования на плоскости
    iX = 0
    iY = 1
    if aTrajectory == None: return None
    for t in aTrajectory:
        t[iX] = t[iX] * multX + sumX
        t[iY] = t[iY] * multY + sumY
    return aTrajectory


def pipe_atma(a_qliq_sm3day, a_fw_perca_, a_length_m, a_pcalc_atma, a_calc_along_flow, aPVTdic, a_theta_deg, a_d_mm,
              a_hydr_corr, a_t_calc_C, a_Tother_C, a_roughness_m=0.0001):
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
    habs_curve_m = length_m * abs(ma.sin_(a_theta_deg / 180 * math.pi))

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
    pCalc = calc_types.paramCalc(aCorrelation=aPVTdic['PVTcorr'], aCalcAlongCoord=False, aFlowAlongCoord=False)
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
        dp_dl = calc_func.calc_grad(x, y[0], a_t_calc_C, pCalc, d_m, a_theta_deg, angle_init_deg, rough_m,
                                    a_qliq_sm3day, a_fw_perca_, aPVTdic, calc_dtdl)
        dy[0] = dp_dl
        #print(f'y = {y}, x = {x}, dy[0] = {dy[0]}')
        return

    s = xalglib.odesolverrkck(y, x, eps, h)
    xalglib.odesolversolve(s, ode_function_1_diff)
    m, xtbl, ytbl, rep = xalglib.odesolverresults(s)

    return ytbl[m-1]

def pipe_atma_pvtcls(a_qliq_sm3day, a_fw_perca_, a_length_m, a_pcalc_atma, a_calc_along_flow, class_pvt, a_theta_deg, a_d_mm,
              a_hydr_corr, a_t_calc_C, a_Tother_C, a_roughness_m=0.0001):
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
    habs_curve_m = length_m * abs(ma.sin_(a_theta_deg / 180 * math.pi))

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
    pCalc = calc_types.paramCalc(aCorrelation=class_pvt.PVTCorr, aCalcAlongCoord=False, aFlowAlongCoord=False)
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
        dp_dl = calc_func.calc_grad_pvtcls(x, y[0], a_t_calc_C, pCalc, d_m, a_theta_deg, angle_init_deg, rough_m,
                                    a_qliq_sm3day, a_fw_perca_, class_pvt, calc_dtdl)
        dy[0] = dp_dl
        #print(f'y = {y}, x = {x}, dy[0] = {dy[0]}')
        return

    s = xalglib.odesolverrkck(y, x, eps, h)
    xalglib.odesolversolve(s, ode_function_1_diff)
    m, xtbl, ytbl, rep = xalglib.odesolverresults(s)

    return ytbl[m-1]