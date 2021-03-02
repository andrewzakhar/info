import pandas as pd
import numpy as np
import xlwings as xw
import os
import scipy.interpolate
from IPython import get_ipython

from scipy.optimize import minimize
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
import matplotlib.dates
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

import dateparser
import statistics
import math

import calc_well_param.calc_func
import calc_well_param.ccalc_pvt
import calc_well_param.cpipe
import calc_well_param.cesp

from common import conf
from common import db_connector
from readers import reader_mf as mf
from readers import reader_kolon
from readers import reader_incl
from readers import reader_pvt
from readers import reader_tr
from readers import reader_tm
from readers import reader_tr

from datetime import datetime
from datetime import timedelta
from writers import writer_bd_vr

config = conf.conf_bd(None, None)

# Настройка папок
# Рабочей папкой считаем папку выше \Скрипт
cCurrentPath_ = os.path.split(os.path.abspath(''))[0]
print('cCurrentPath_ =', cCurrentPath_)
# cWorkFolder = r'F:\Work'
cWorkFolder = cCurrentPath_
print(cWorkFolder)

def create_power_from_production_table_py(esp, Freq_, mu_):
    # Для создания табличной функции дебита от мощности.

    production = [5, 10]

    esp.mu_cSt = float(mu_)
    esp.correct_visc = True
    esp.freq_hz = float(Freq_)
    power_w = [esp.esp_power_w(aqliq_m3day=production[0]) / 1000, esp.esp_power_w(aqliq_m3day=production[-1]) / 1000]

    i = 1
    while power_w[i] != power_w[i-1] and production[-1] < 1000: #production[-1] < np.floor(Mean_prod_ * 1.2):
        production.append(production[-1] + 5)
        power_calc = esp.esp_power_w(aqliq_m3day=production[-1])
        if power_calc == None:
            power_w.append(power_w[-1])
        else:
            power_w.append(power_calc / 1000)
        i += 1
    if len(power_w) > 2:
        del power_w[-1]
        del production[-1]

    diff = np.gradient(power_w, production)
    df = pd.DataFrame(columns = ['Production', 'power_w', 'Derivative'])
    df['Production'] = production
    df['power_w'] = power_w
    df['Derivative'] = diff
    return df

def plot_pump(passport_power_table, well_name, Freq_, mu_, aPath):
    # Функция построения напорно-расходной хар-ки по мощности

    fig, ax = plt.subplots(figsize=(15, 15))
    plt.tight_layout()

    ax.plot(passport_power_table['Production'], passport_power_table['power_w'], color='blue', label = 'Мощность от производительности')
    ax.set_xlabel("Дебит",fontsize=14)
    ax.set_ylabel("Мощность",fontsize=14)
    ax.legend()
    ax.set_title(f'{field_name} {well_name} Напорно-расходная характеристика насоса по мощности (F = {Freq_} Гц, вязкость {mu_})')

    ax2=ax.twinx()
    ax2.plot(passport_power_table['Production'], passport_power_table['Derivative'], color='red', label = 'Производная')
    ax2.set_ylabel("Derivative",fontsize=14)
    ax2.legend()

    aFileName = field_name + ' ' + well_name + '_насоса_хар.jpg'
    plt.savefig(os.path.join(aPath, aFileName), quality=100)
    plt.close()


def get_Q_prediction(low, df, First_liq_point_, esp, Freq_start_, PVTdic, passport_power_table, Fw_mean_, well_name, aPath):
    # Предсказание через напорно-расходную хар-ку
    q_predicted_array = [] # Массив предсказанных дебитов в условиях насоса
    q_liq_surface = [] # Массив предсказанных дебитов в поверхностных условиях
    counter_off_bounds = 0 # Счетчик точек, не попавших на паспортную хар-ку
    mu_array = [] # Массив вязкостей
    max_dQ_array = [] # Массив максимальных допустимых изменений дебита с учетом напорной хар-ки на текущем шаге
    mu_ = PVTdic['muob_cP']

    for i in range(len(df)):

        ajusted_flag = False # Указывает, варьировалась ли вязкость, если true, то нужно перестраивать напорно-расходную

        if i != 0:
            mu_temp = mu_ # Параметр для варьирования вязкости
            Freq_ = df['F ESP'].iloc[i]
            #Fw_ = df['Fw'].iloc[i]

            #mu_ = int(include_emulsion(Fw_))

            if Freq_ != Freq_start_:
                Freq_start_ = Freq_
                passport_power_table = create_power_from_production_table_py(esp, Freq_start_, mu_)

                #plot_pump(passport_power_table, well_name, Freq_start_, mu_, aPath)

                #print('new pass because of another frequency')



            #Q_old = interpolated_power(df['Expected power'].iloc[i] # Значение дебита на прошлом шаге
            N_passport = passport_power_table['power_w'].values
            Q_passport = passport_power_table['Production'].values

            Q_result = np.empty(0)
            Q_for_interpolating = []
            N_for_interpolating = []
            n_new = df['Expected power'].iloc[i]
            if low == True:
                Q_old = df['Q mix pump cond'].iloc[i-1]
            else:
                Q_old = q_predicted_array[i-1]

            for j in range(1, len(N_passport)):
                if (N_passport[j-1] <= n_new and N_passport[j] >= n_new) or (
                    N_passport[j-1] >= n_new and N_passport[j] <= n_new):

                    N_for_interpolating = [N_passport[j-1], N_passport[j]]
                    Q_for_interpolating = [Q_passport[j-1], Q_passport[j]]
                    interpolated_power = scipy.interpolate.interp1d(N_for_interpolating, Q_for_interpolating)
                    Q_result = np.append(Q_result, interpolated_power(n_new))

            print('Possible Q is', Q_result)

            if len(Q_result) != 0:
                if Q_old == 0:
                    #Если модель ушла в 0, то опираемся на последнее ненулевое значение
                    Q_old = q_predicted_array[[i for i, e in enumerate(q_predicted_array) if e != 0][-1]]
                Q = Q_result.flat[np.abs(Q_result - Q_old).argmin()] #Предсказываем дебит ближайший к последнему != 0
                print('Selecting Q= ', Q)
            else:
                Q = 0
                print('Point out of model, Q = ', Q)
                counter_off_bounds += 1
            #interpolated_power = scipy.interpolate.interp1d(
            #    passport_power_table['Power'].values, passport_power_table['Production'].values,
            #    bounds_error=True)#"extrapolate")

            print('Expected power is {}'.format(df['Expected power'].iloc[i]))
            try:
                print('Real power is {}'.format(df['Active power'].iloc[i]))
            except:
                print('Real power is {}'.format(df['Active power 1H'].iloc[i]))
            print('Frequency is {}'.format(Freq_))

            if df['Expected power'].iloc[i] < 10 or Freq_ < 10:
                print('Pump is off!')
                Q = 0

        Max_deriv_ = abs(max(passport_power_table['Derivative'].values)) # Максимальное зн-е производной N'q
        dN_ = abs(df['dN expected'].iloc[i])
        Max_dQ = abs(dN_ / Max_deriv_) # Максимально возможное изменение дебита

        max_dQ_array.append(Max_dQ)

        if i == 0:
            Q = First_liq_point_ # Доверяем первому замеру

        q_predicted_array.append(Q)

        #!!!!
        #Дебит на поверхности

        q_liq_surface.append(transform_to_surface_conditions(Q, PVTdic, i, df))
        #!!!!

        #print('Q pump is {}'.format(Q))
        #print('Expected power is ', df['Expected power'].iloc[i])
        #print(passport_power_table)

        if ajusted_flag == True:
            passport_power_table = create_power_from_production_table_py(esp, Freq_start_, mu_)

        print('Date is {}, Q predicted is {}, progress is {}%'.format(df.index[i], Q, np.round(i/len(df) * 100, 2)))

    if low == False: #чтобы выводить только для 3-х часового предсказания
        print('Total number of points out of interpolation is {}'.format(counter_off_bounds))
        #total_off_bounds = counter_off_bounds

    return (q_predicted_array, counter_off_bounds, mu_array, max_dQ_array, q_liq_surface)


def transform_to_surface_conditions(Q_pump, PVTdic, i, df):

    Q_start = 100  # Стартовая точка оптимизации

    minimized_result = minimize(get_error_for_optimization, Q_start, method='Nelder-Mead',
                                args=[i, df, Q_pump, PVTdic])  # Запуск оптимизатора
    print('True value is {}, predicted is {}, error is {}'.format(Q_pump, minimized_result.x[0],
                                                                  np.abs(Q_pump - minimized_result.x[0])/Q_pump))
    # print(minimized_result)
    print('{}% completed'.format(np.round(i / len(df) * 100, 2)))

    return minimized_result.x[0]

def get_error_for_optimization(Q, args):

    i = args[0] # Текущая точка в предсказанном массиве
    df = args[1] # ДФ со всеми данными

    Q_true = args[2] #Реальный дебит модели в условиях насоса
    PVTdic = args[3]
    Fw_ = df['Fw'].iloc[i]
    #Temperature = df['Temperature on the surface'].iloc[i]
    PVTdic['pksep_atma'] = df['PINP'].iloc[i]
    PVTdic['tksep_C'] = df['PED_T'].iloc[i]
    PVTdic['rp_m3m3'] = df['Газожидкостной фактор (рассчитанный)'].iloc[i]
    z_class = calc_well_param.calc_func.z_factor_2015_kareem()
    z = z_class.z_factor
    PVTdic = calc_well_param.ccalc_pvt.calc_pvt_vr(PVTdic['pksep_atma'], PVTdic['tksep_C'], Q, Fw_ / 100, PVTdic['PVTcorr'], PVTdic,
                                                   z)

#    PVTdic = calc_well_param.ccalc_pvt.calc_pvt_vr(PVTdic['pksep_atma'], PVTdic['tksep_C'], Q, Fw_ / 100, PVTdic['PVTcorr'], PVTdic,
#                                                   calc_well_param.calc_func.factor_2015_kareem)

    if 'q_mix_rc_m3day' in PVTdic:
        Q_predicted = PVTdic['q_mix_rc_m3day']
    else:
        PVTdic.update({'q_mix_rc_m3day': 0})

    error = math.sqrt((Q_predicted - Q_true)**2)

    return error

def reports(summary_df_xls):
    summary_df_xls.rename(columns={'Active power 1H': 'Активная мощность (ТМ)', 'F ESP': 'F Гц насоса (ТМ)',
                                  'K degradation': 'К деград', 'PED_T': 'ПЭД Т град (ТМ)', 'PLIN': 'P лин (ТМ)',
                                  'PINP': 'Давление на входе ЭЦН (ТМ)',
                                  'Expected power': 'Мощность насоса ожид',
                                  'Expected production': 'Qж (ТМ) в усл.насоса ожид',
                                  'Expected production on the surface': 'Qж (ТМ) в поверх.услов. ожид',
                                   'LIQ_RATE': 'Qж (ТМ)'},
                         inplace=True)

    summary_df_xls.to_excel(os.path.join(aPathPassport, 'summary 1H.xls'))

    # Построение графиков

    # Сравнение входного Qж (ТМ) и расчетного в поверхностных условиях
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.tight_layout()

    ax.scatter(summary_df_xls.index, summary_df_xls['Qж (ТМ)'], color='blue', label = 'Qж (ТМ)')
    ax.plot(summary_df_xls.index, summary_df_xls['Qж (ТМ) в усл.насоса ожид'], color='red', label = 'Qж (ТМ) в усл.насоса ожид')
    ax.plot(summary_df_xls.index, summary_df_xls['Qж (ТМ) в поверх.услов. ожид'], color='green', label = 'Qж (ТМ) в поверх.услов. ожид')

    ax.legend()
    ax.set_title(f'{field_name} {well_name} Сравнение входного Qж (ТМ) и расчетного в поверхностных условиях ({summary_df_xls.index[0]} - {summary_df_xls.index[-1]})')

    aFileName = field_name + ' ' + well_name + '_qliq_expected_vs_tm.jpg'
    plt.savefig(os.path.join(aPathPassport, aFileName), quality=100)
    plt.close()

    # Сравнение входного Qж (ТМ) и расчетного в поверхностных условиях
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.tight_layout()

    ax.scatter(summary_df_xls.index, summary_df_xls['Активная мощность (ТМ)'], color='blue', label = 'Активная мощность (ТМ)')
    ax.plot(summary_df_xls.index, summary_df_xls['Мощность насоса ожид'], color='red', label = 'Мощность насоса расчетная')

    ax.legend()
    ax.set_title(f'{field_name} {well_name} Сравнение Активной мощности (ТМ) и расчетной ({summary_df_xls.index[0]} - {summary_df_xls.index[-1]})')

    aFileName = field_name + ' '+ well_name + '_activ_power_expected.jpg'
    plt.savefig(os.path.join(aPathPassport, aFileName), quality=100)
    plt.close()

def pvt_export(pvt, df_tm, aPathPassport):
    pvt_index = ['Источники', 'Единиц.изм', 'Значения']

    summary_df_pvt = pd.DataFrame(index=pvt_index, columns=['Плотность газа попутного относительная',
                                                            'Удельный вес сепарированной нефти',
                                                            'Удельный вес попутной воды',
                                                            'Давление насыщения нефти газом',
                                                            'Температура пласта',
                                                            'Объемный коэффициент нефти',
                                                            'Динамическая вязкость сепарированной нефти',
                                                            'Газосодержание, м3/м3, станд.сепар.',
                                                            'Газовый фактоp общий'
                                                            ])
    summary_df_pvt['Плотность газа попутного относительная'] = ['sppl_sk.pg_1', '', pvt.gamma_gas]
    summary_df_pvt['Удельный вес сепарированной нефти'] = ['sppl_sk.pn_1', 'т/м3', pvt.gamma_oil]
    summary_df_pvt['Удельный вес попутной воды'] = ['sppl_sk.pv_1', 'т/м3', pvt.gamma_wat]
    summary_df_pvt['Давление насыщения нефти газом'] = ['sppl_sk.dn_1', 'атм', pvt.pb_atma]
    summary_df_pvt['Температура пласта'] = ['sppl_sk.tl_1', 'град С', pvt.tres_C]
    summary_df_pvt['Объемный коэффициент нефти'] = ['sppl_sk.ok_1', '', pvt.bob_m3m3]
    summary_df_pvt['Динамическая вязкость сепарированной нефти'] = ['sppl_sk.vn_1', 'спз', pvt.muob_cP]
    summary_df_pvt['Газосодержание, м3/м3, станд.сепар.'] = ['Константа', 'м3/м3', pvt.rsb_m3m3]
    summary_df_pvt['Газовый фактоp общий'] = ['sppl_sk.fo_1', 'нм3/т', pvt.rp_m3m3]

    summary_df_pvt.to_excel(os.path.join(aPathPassport, 'pvt.xls'))

    df_tm.to_excel(os.path.join(aPathPassport, 'tm_filter.xls'))


param_id = 12
Dintake_ = 100 # Диаметр приемной сетки насоса (пока что одинаковый для всех)
KsepGasSep_ = 0.7 # Коэффициент сепарации
#TKsep_ = 89 # Температура сепарации
#Tintake_ = 20 # Температура на приеме
PVT_corr_ = 0 # PVT корреляция для записи в строку PVT

PVTdic = {}
PVTdic['PVTcorr'] = PVT_corr_
PVTdic['ksep_fr'] = float(KsepGasSep_)
PVTdic['qgas_free_sm3day'] = 0

# PVTdic['bwSC_m3m3'] = const.pvt_fields[field_code]['bwSC_m3m3']
# PVTdic['salinity_ppm'] = calc_well_param.ccalc_pvt.unf_calc_Sal_BwSC_ppm(PVTdic['bwSC_m3m3'])

df_well_list = pd.DataFrame(columns=['calc_start', 'calc_end']) # , 2860198200, 6130092100, 6130094600, 6130097500, 6110273800, 6110246600,
df_well_list.index.name = 'well_id'
df_well_list.loc[2860198200] = [datetime(2020, 1, 5), datetime(2020, 2, 5)]
df_well_list.loc[2860081800] = [datetime(2020, 4, 25), datetime(2020, 5, 25)]
df_well_list.loc[6110130800] = [datetime(2020, 4, 25), datetime(2020, 5, 25)]
df_well_list.loc[6130094600] = [datetime(2020, 4, 25), datetime(2020, 5, 25)]
df_well_list.loc[6130097500] = [datetime(2020, 4, 25), datetime(2020, 5, 25)]
df_well_list.loc[6110273800] = [datetime(2020, 4, 25), datetime(2020, 5, 25)]
df_well_list.loc[6110246600] = [datetime(2020, 4, 25), datetime(2020, 5, 25)]
df_well_list.loc[6110258600] = [datetime(2020, 4, 25), datetime(2020, 5, 25)]
df_well_list.loc[2860197600] = [datetime(2020, 4, 25), datetime(2020, 5, 25)]
df_well_list.loc[2370000800] = [datetime(2020, 4, 25), datetime(2020, 5, 25)]
df_well_list.loc[2550601200] = [datetime(2020, 4, 25), datetime(2020, 5, 25)]
#df_well_list.loc[2860193500] = [datetime(2020, 4, 25), datetime(2020, 5, 25)] нет параметра Глубина спуска насоса (esp.pump_depth_m)




# 2860197600,
# 2860065000 - Ошибка на инструкции  df_res['K degradation'].iloc[-1] = df_res['K degradation'].iloc[-2] 2860081800, 6110206400, 2860193500, 6110114200, 6130092100
# 6130117800 - Ошибка     HwBEP_ft = tmp_esp.get_esp_head_m(self.nom_rate_m3day) * const.const_convert_m_ft for *: 'NoneType' and 'float'
# well_list = [2860198200, 2860193500, 2860197600, 2860048400, 2860065000, 2860081800,
#
# Посчитано 6110130800, 2860198200, 6130094600, 6130097500, 6110273800, 6110246600, 6110258600, 2860197600, 2370000800
# Нет Рприем 6130103400, 6130105000,
# Нет Акт.Мощ 6130212800, 6130214800
# Сам выбрал для теста

# Количество скважин для расчета, считаем по порядку по well_list
# calc_count = 1

N = 1  # Параметр для проверки качества модели (выкидываем каждую N-ю точку и считаем ошибку)

## Прочитаем Мехфонд по всем скважинам
reader = mf.reader_mf(config, df_well_list.index.tolist())
df = reader.df_source_data

## Прочитаем Инклиметрию по всем скважинам
reader_incl = reader_incl.reader_incl_ora(config, df_well_list.index.tolist())

well_list = df_well_list.index.tolist()
for well_id in df_well_list.index:

    calc_start = df_well_list['calc_start'].loc[well_id]
    calc_end = df_well_list['calc_end'].loc[well_id]

    calc_time_start = datetime.now()

    reader_pvt_wells = reader_pvt.reader_pvt_ora(config, df_well_list.index.tolist(), calc_start)
    pvt = reader_pvt_wells.get_pvt(well_id)
    field_id = reader_pvt_wells.df_source_data.loc[[well_id], ['FIELD_CODE']].iloc[0].FIELD_CODE
    field_name = reader_pvt_wells.df_source_data.loc[[well_id], ['FIELD_NAME']].iloc[0].FIELD_NAME
    well_name = reader_pvt_wells.df_source_data.loc[[well_id], ['WELL_NAME']].iloc[0].WELL_NAME
    # ТР читаем на конец периода, чтобы взять последний ТР
    reader_tr_well = reader_tr.reader_tr_ora(config, well_id, calc_end)
    water_cut_tr = reader_tr_well.df_source_data.loc[[well_id], ['WATER_CUT']].iloc[0].WATER_CUT
    oil_rate_tr = reader_tr_well.df_source_data.loc[[well_id], ['OIL_RATE']].iloc[0].OIL_RATE
    qgas_tr = reader_tr_well.df_source_data.loc[[well_id], ['QGAS']].iloc[0].QGAS

    print(f'Текущая скважина для расчета: {well_name} - {field_name} ({well_id})')

    aPathPassport = os.path.join(cWorkFolder,
                                    r'Данные для виртуальной расходометрии\БД', field_name, well_name)
    if not os.path.exists(aPathPassport):
        os.makedirs(aPathPassport)

    PVTdic['gamma_gas'] = pvt.gamma_gas
    PVTdic['gamma_oil'] = pvt.gamma_oil
    PVTdic['gamma_wat'] = pvt.gamma_wat
    PVTdic['rsb_m3m3'] = pvt.rsb_m3m3
    PVTdic['pb_atma'] = pvt.pb_atma
    PVTdic['tres_C'] = pvt.tres_C
    PVTdic['bob_m3m3'] = pvt.bob_m3m3
    PVTdic['muob_cP'] = pvt.muob_cP
    PVTdic['bwSC_m3m3'] = pvt.bwSC_m3m3
    PVTdic['salinity_ppm'] = calc_well_param.ccalc_pvt.unf_calc_Sal_BwSC_ppm(pvt.bwSC_m3m3)

    log_df_py = pd.DataFrame(index=well_list, columns=['ESP name real', 'ESP name used', 'ESP id',
                                                      'Average production', 'MAE score', 'Average error, %',
                                                      'Maximum absolute error', 'Average production surface',
                                                      'MAE score surface', 'Average error surface, %',
                                                      'Maximum absolute error surface', 'MAE score surface restoring',
                                                      'Maximum absolute error surface restoring',
                                                      'Average error because incorrect passport, %',
                                                      'Total № of points off passport bounds', 'R2 score',
                                                      'R2 test score ridge', 'R2 train score ridge',
                                                      'MAE test score ridge', 'MAE train score ridge',
                                                      'Average MAE test score ridge, %',
                                                      'Average MAE train score ridge, %', 'R2 test score forest',
                                                      'R2 train score forest', 'MAE test score forest',
                                                      'MAE train score forest', 'Average MAE test score forest, %',
                                                      'Average MAE train score forest, %'])

    esp = reader.get_esp(well_id);

    if not esp.pump_depth_m:
        print(f'Расчет остановлен: нет параметра Глубина спуска насоса (esp.pump_depth_m)')
        continue
    ESP_name_unifloc_py = esp.pump_name
    ESP_manufacturer_py = esp.manufacturer_name
    ESP_nom_rate_py = esp.nom_rate_m3day    # 59
    ESP_stage_num_py = esp.stage_num    # ступеней макс 522
    ESP_max_rate_m3day_py = esp.max_rate_m3day # 124

    reader_kolon_bd = reader_kolon.reader_kolon_ora(config, df_well_list.index.tolist())

    reader_tm_well = reader_tm.reader_tm_ora(config, well_id)
    all_data_high = reader_tm_well.get_data(calc_start, calc_end)

    #
    df_tm = pd.DataFrame(index=['NULL', 'Ноль', 'Новое значение', 'Комментарий', '1', '2', '3', '4'],
                         columns=['Обводненность', 'Частота', 'Qн', 'Qгп', 'Температ.двиг'])
    # Проверка и заполнение Обводненности
    null_sum = all_data_high['WATER_CUT'].isnull().sum()
    if null_sum > 0:
        all_data_high['WATER_CUT'].fillna(value=water_cut_tr, inplace=True) # Подменить пустышки из ТР
        df_tm.loc['NULL', 'Обводненность'] = f'Есть, замена (всего {null_sum})'
        df_tm.loc['Новое значение', 'Обводненность'] = water_cut_tr
    nol_sum = sum(all_data_high['WATER_CUT'] == 0)
    if nol_sum > 0:
        all_data_high.loc[all_data_high['WATER_CUT'] == 0, 'WATER_CUT'] = water_cut_tr
        df_tm.loc['Ноль', 'Обводненность'] = f'Есть, замена (всего {nol_sum})'
        df_tm.loc['Новое значение', 'Обводненность'] = water_cut_tr

    # Проверка и заполнение Частоты
    null_sum = all_data_high['FREQ_HZ'].isnull().sum()
    if null_sum > 0:
        all_data_high['FREQ_HZ'].fillna(value=calc_well_param.const.esp_freq_hz_default, inplace=True) # Подменить пустышки из ТР
        df_tm.loc['NULL', 'Частота'] = f'Есть, замена (всего {null_sum})'
        df_tm.loc['Новое значение', 'Частота'] = calc_well_param.const.esp_freq_hz_default
    nol_sum = sum(all_data_high['FREQ_HZ'] == 0)
    if nol_sum > 0:
        all_data_high.loc[all_data_high['FREQ_HZ'] == 0, 'FREQ_HZ'] = calc_well_param.const.esp_freq_hz_default

    # Проверка и заполнение Qн
    null_sum = all_data_high['OIL_RATE'].isnull().sum()
    if null_sum > 0:
        all_data_high['OIL_RATE'].fillna(value=oil_rate_tr, inplace=True) # Подменить пустышки из ТР
        df_tm.loc['NULL', 'Qн'] = f'Есть, замена (всего {null_sum})'
        df_tm.loc['Новое значение', 'Qн'] = oil_rate_tr
    nol_sum = sum(all_data_high['OIL_RATE'] == 0)
    if nol_sum > 0:
        all_data_high.loc[all_data_high['OIL_RATE'] == 0, 'OIL_RATE'] = oil_rate_tr

    # Проверка и заполнение Qгп
    null_sum = all_data_high['QGAS'].isnull().sum()
    if null_sum > 0:
        all_data_high['QGAS'].fillna(value=qgas_tr, inplace=True) # Подменить пустышки из ТР
        df_tm.loc['NULL', 'Qгп'] = f'Есть, замена (всего {null_sum})'
        df_tm.loc['Новое значение', 'Qгп'] = qgas_tr
    nol_sum = sum(all_data_high['QGAS'] == 0)
    if nol_sum > 0:
        all_data_high.loc[all_data_high['QGAS'] == 0, 'QGAS'] = qgas_tr

    # Проверка и заполнение Температуры двигателя/жидкости
    null_sum = all_data_high['PED_T'].isnull().sum()
    if null_sum > 0:
        all_data_high['PED_T'].fillna(value=pvt.tres_C, inplace=True) # Подменить пустышки из PVT
        df_tm.loc['NULL', 'Температ.двиг'] = f'Есть, замена (всего {null_sum})'
        df_tm.loc['Новое значение', 'Температ.двиг'] = pvt.tres_C
    nol_sum = sum(all_data_high['PED_T'] == 0)
    if nol_sum > 0:
        all_data_high.loc[all_data_high['PED_T'] == 0, 'PED_T'] = pvt.tres_C

    df_tm.loc['Комментарий', 'Обводненность'] = 'Обводненность -> Обводненность из ТР'
    df_tm.loc['1', 'Обводненность'] = 'Частота -> стандартная 50 Гц'
    df_tm.loc['2', 'Обводненность'] = 'Qн  -> ТР ШТР'
    df_tm.loc['3', 'Обводненность'] = 'Qгп -> ТР ШТР'
    df_tm.loc['4', 'Обводненность'] = 'Температатура ПЭД -> Температура пласта из PVT'

    all_data_high['Газовый фактор (рассчитанный)'] = all_data_high['QGAS'] / all_data_high['OIL_RATE']
    all_data_high['Газожидкостной фактор (рассчитанный)'] = all_data_high['QGAS'] / all_data_high['LIQ_RATE']
    # Чистим пустышки
    all_data = all_data_high.dropna()
    # Убираем строки с нулевым значеними
    all_data = all_data.drop(all_data[(all_data['PINP'] == 0) |
                  (all_data['PLIN'] == 0) | (all_data['ACTIV_POWER'] == 0) |
                  (all_data['LIQ_RATE'] == 0)].index)

    pvt_export(pvt, df_tm, aPathPassport)

    #  (all_data['FREQ_HZ'] == 0) |
    #

    all_data_high['WATER_CUT'].interpolate(limit_direction='both', inplace=True)

    is_continue = False
    if len(all_data.index) == 0:
        print('Неверные входные данные - Нет ТМ по Qж, Рприем, Активной мощности')
        is_continue = True
    if esp.esp_max_rate_m3day != esp.esp_max_rate_m3day or not esp.esp_max_rate_m3day or \
            esp.esp_nom_rate_m3day != esp.esp_nom_rate_m3day or not esp.esp_nom_rate_m3day:
        print('Неверные входные данные - неполные данные по насосу (Номинальный или максимальный дебит)')
        is_continue = True
    if is_continue:
        continue

    if esp.stage_num != esp.stage_num or not esp.stage_num:
        print('Неверные входные данные - неполные данные по насосу (Количество ступеней)')
        is_continue = True
    if esp.head != esp.head or not esp.head:
        print('Неверные входные данные - неполные данные по насосу (Напор)')
        is_continue = True


    if is_continue:
        print('Неверные входные данные - пропуск расчета')
        continue
    all_data['Газожидкостной фактор (рассчитанный)'].interpolate(limit_direction='both', inplace=True)
    all_data['FREQ_HZ'].interpolate(limit_direction='both', inplace=True)
    all_data['PED_T'].interpolate(limit_direction='both', inplace=True)

    if all_data.index[0] < all_data_high.index[0]:
        all_data = all_data.truncate(before=all_data_high.index[0])
        all_data_high = all_data_high.truncate(before=all_data.index[0])
    else:
        all_data_high = all_data_high.truncate(before=all_data.index[0])
        all_data = all_data.truncate(before=all_data_high.index[0])

    if all_data.index[-1] > all_data_high.index[-1]:
        all_data = all_data.truncate(after=all_data_high.index[-1])
        all_data_high = all_data_high.truncate(after=all_data.index[-1])
    else:
        all_data_high = all_data_high.truncate(after=all_data.index[-1])
        all_data = all_data.truncate(after=all_data_high.index[-1])

    print('Input data begins at {}, {}, ends at {}, {}'.format(all_data_high.index[0], all_data.index[0],
                                                   all_data_high.index[-1], all_data.index[-1]))
    # Проверка на Мпа/атм
    print('Average pressure, ', all_data['PLIN'].dropna().mean())
    if all_data['PLIN'].dropna().mean() < 3:
        all_data['PLIN'] = all_data['PLIN'] * 9.869
        all_data_high['PLIN'] = all_data_high['PLIN'] * 9.869

    active_power_1H = all_data_high[['ACTIV_POWER']]
    f_esp_1H = all_data_high[['FREQ_HZ']]
    p_lin_1H = all_data_high[['PLIN']]
    tKsep_1H = all_data_high[['PED_T']]
    pksep_atma_1H = all_data_high[['PINP']]
    rp_1H = all_data_high[['Газожидкостной фактор (рассчитанный)']]
    wc_1h = all_data_high[['WATER_CUT']]
    qliq_1H = all_data_high[['LIQ_RATE']]

    f_esp_1H = f_esp_1H.rename(columns={'FREQ_HZ': 'F ESP'})

    reader_incl.get_depths_list(well_id)
    inclinometry = reader_incl.depth_angle_df

    pressure_at_the_discharge_py = []

    k_deg = []  # Массив суточных коэффициентов деградации

    passp_pow_py = []  # Массив паспортных мощностей раз в сутки по результатам расчета python
    q_mix_array_py = []  # Массив суточных дебитов в условиях насоса по результатам расчета python

    Freq_start_ = all_data['FREQ_HZ'].iloc[0]  # частота ЭЦН в первый день
    df_res = pd.DataFrame(index=all_data.index, columns=['Q mix pump cond', 'K degradation'])

    for i in range(len(all_data)):
        Pksep_atma_ = float(all_data['PINP'].iloc[i])
        if Pksep_atma_ is None or Pksep_atma_ != Pksep_atma_:
            Pksep_atma_ = 0
        Rp_ = all_data['Газожидкостной фактор (рассчитанный)'].iloc[i]
        Qliq_ = all_data['LIQ_RATE'].iloc[i]
        Fw_ = all_data['WATER_CUT'].iloc[i]
        Fw_mean_ = all_data[['WATER_CUT']].mean()  # [0]
        Active_pow_ = all_data['ACTIV_POWER'].iloc[i]

        # mu_ = int(include_emulsion(Fw_mean_))
        # mu_ = int(include_emulsion(Fw_))

        Pintake_ = Pksep_atma_
        Plin_ = all_data['PLIN'].iloc[i]
        Freq_ = float(all_data['FREQ_HZ'].iloc[i])
        TKsep_ = all_data['PED_T'].iloc[i]
        Tintake_ = TKsep_

        print(f' Дата: {all_data.index[i]}, Pksep_atma_: {Pksep_atma_}, Qliq_: {Qliq_}, Rp_: {Rp_}')

        PVTdic['pksep_atma'] = float(Pksep_atma_)
        PVTdic['tksep_C'] = float(TKsep_)
        PVTdic['rp_m3m3'] = float(Rp_)

        pressure_full_temp_py = [Plin_]  # Распределение давления по стволу (свое в каждый день)

        j = 1
        while inclinometry.iloc[j]['DEPTH'] < esp.pump_depth_m:
            Length_ = inclinometry.iloc[j]['DEPTH'] - inclinometry.iloc[j - 1]['DEPTH']
            depth = inclinometry.iloc[j]['DEPTH']
            Dtub_ = reader_kolon_bd.get_ekolon_row(well_id, depth)

            #print(f'         Глубина: {depth}, Dtub_: {Dtub_}, j = {j}')

            Pcalc_ = pressure_full_temp_py[j - 1]
            Calc_along_flow_ = 0
            Theta_deg = inclinometry.iloc[j]['ANGLE']
            Hydr_corr_ = 0
            Tcalc_ = PVTdic['tres_C']
            Tother_ = Tcalc_

            if Pksep_atma_ == Pksep_atma_ and Qliq_ == Qliq_ and Fw_ == Fw_:
                pressure_py = calc_well_param.cpipe.pipe_atma(Qliq_, Fw_, float(Length_), Pcalc_, Calc_along_flow_,
                                                              PVTdic, Theta_deg, Dtub_, Hydr_corr_, Tcalc_, Tother_)
                if not pressure_py or not pressure_py[0]:
                    print('Расчет давления = None')
                    pressure_full_temp_py.append(0)
                else:
                    pressure_full_temp_py.append(pressure_py[0])
            else:
                pressure_full_temp_py.append(0)

            # print(f'    depth = {depth}, pressure_py = {pressure_py}, Length_ = {Length_}, Pcalc_ = {Pcalc_}, Theta_deg = {Theta_deg}')
            j += 1
        pass
        z_class = calc_well_param.calc_func.z_factor_2015_kareem()
        z = z_class.z_factor
        if Pksep_atma_ == Pksep_atma_ and Qliq_ == Qliq_ and Fw_ == Fw_:
            PVTdic = calc_well_param.ccalc_pvt.calc_pvt_vr(Pintake_, Tintake_, Qliq_, Fw_ / 100, PVT_corr_, PVTdic,
                                                           z)

        #        if Pksep_atma_ == Pksep_atma_ and Qliq_ == Qliq_ and Fw_ == Fw_:
        #            PVTdic = calc_well_param.ccalc_pvt.calc_pvt_vr(Pintake_, Tintake_, Qliq_, Fw_ / 100, PVT_corr_, PVTdic,
        #                                                           calc_well_param.calc_func.factor_2015_kareem)

        if 'q_mix_rc_m3day' in PVTdic:
            Q_mix_intake_py = PVTdic['q_mix_rc_m3day']
        else:
            PVTdic.update({'q_mix_rc_m3day': 0})
        pressure_at_the_discharge_py.append(pressure_full_temp_py[-1])
        # print('Unifloc Q_mix_intake_ =', Q_mix_intake_, 'py Q_mix_intake_py =', Q_mix_intake_py)
        #print('py pressure_full_temp_py[-1] =', pressure_full_temp_py[-1], 'Q_mix_intake_py =', Q_mix_intake_py)

        PVTdic = calc_well_param.ccalc_pvt.calc_pvt_vr(pressure_at_the_discharge_py[-1], Tintake_, Qliq_, Fw_ / 100,
                                                       PVT_corr_, PVTdic, z)

#        PVTdic = calc_well_param.ccalc_pvt.calc_pvt_vr(pressure_at_the_discharge_py[-1], Tintake_, Qliq_, Fw_ / 100,
#                                                       PVT_corr_, PVTdic, calc_well_param.calc_func.factor_2015_kareem)
        Q_mix_discharge_py = PVTdic['q_mix_rc_m3day']
        Q_mix_py = (Q_mix_intake_py + Q_mix_discharge_py) / 2  # Дебит ГЖС в условиях насоса
        # print('Unifloc Q_mix_ =', Q_mix_, 'py Q_mix_py =', Q_mix_py)
        q_mix_array_py.append(Q_mix_py)


        esp.mu_cSt = PVTdic['muob_cP']
        esp.correct_visc = True
        esp.freq_hz = float(Freq_)
        power_w = esp.esp_power_w(aqliq_m3day=Q_mix_py) / 1000

        k_deg.append(Active_pow_ / power_w)

        print(f'Active_pow_ {Active_pow_} / power_w {power_w}')
        passp_pow_py.append(power_w)
        pass

    # print(f'q_mix_array_py.count = {len(q_mix_array_py.index)}, k_deg.count = {len(k_deg.index)}')
    df_res['Q mix pump cond'] = q_mix_array_py
    df_res['K degradation'] = k_deg
    df_res['F ESP'] = all_data['FREQ_HZ']
    df_res['Expected power'] = passp_pow_py
    df_res['Active power'] = all_data['ACTIV_POWER']
    df_res['dN expected'] = df_res['Expected power'].diff() # Изменение активной мощности в 8 часовых данных
    df_res['dN expected'].fillna(method='backfill', inplace=True)
    df_res['Газожидкостной фактор (рассчитанный)'] = all_data['Газожидкостной фактор (рассчитанный)']

    df_res['Fw'] = all_data['WATER_CUT']
    df_res['PINP'] = all_data['PINP']
    df_res['PED_T'] = all_data['PED_T']
    df_res['LIQ_RATE'] =  all_data['LIQ_RATE']
    df_res['PLIN'] = all_data['PLIN']

    #Данные за каждые сутки без ошибок в вычислениях
    df_res.drop(df_res[df_res['Q mix pump cond'] == -1].index, inplace=True)
    #df_res[['F ESP']] = df_res[['F ESP']].interpolate() # На случай пропусков в суточных данных
    df_res.dropna(inplace=True) # На случай пропусков в суточных данных

    Mean_prod_py = df_res['Q mix pump cond'].mean() # Средний дебит по скважине (после выброса -1!)

    # Производим расчеты с выбросом некоторых дней с целью контроля ошибки

    df_res['K degradation'].iloc[-1] = df_res['K degradation'].iloc[-2]

    each_N_day = df_res.iloc[::N] # Данные за каждый N-й день
    if N > 1:
        #ДФ с днями, на которых проверяем работу алгоритма (выкидывали каждый N-й день)
        validation_days = df_res.drop(index=each_N_day.index)
    else:
    #Для проверки последнего замера
        validation_days = df_res.iloc[-1:]
        each_N_day.drop(index=validation_days.index, inplace=True)


    resampled_k_deg = df_res[['K degradation']].resample('1H')

    interpolated_k_deg_1H = resampled_k_deg.interpolate('index')

    interpolated_rp = rp_1H.interpolate()

    summary_df = pd.concat([active_power_1H, f_esp_1H, interpolated_k_deg_1H, tKsep_1H, wc_1h, p_lin_1H, interpolated_rp, pksep_atma_1H], axis=1).reindex()

    summary_df.dropna(inplace=True)

    summary_df.drop(summary_df[(summary_df['F ESP'] == 0) | (summary_df['ACTIV_POWER'] == 0) |
                               (summary_df['PINP'] == 0)].index, inplace=True)

    summary_df['Expected power'] = summary_df['ACTIV_POWER'] / summary_df['K degradation']

    summary_df['dN expected'] = summary_df['Expected power'].diff()  # Изменение активной мощности
    summary_df['dN expected'].fillna(method='backfill', inplace=True)

    summary_df.rename(columns={'ACTIV_POWER': 'Active power 1H', 'WATER_CUT': 'Fw'},
                      inplace=True)

    # Приводим к одному моменту начала py

    if df_res.index[0] < summary_df.index[0]:
        df_res = df_res.truncate(before=summary_df.index[0])
        summary_df = summary_df.truncate(before=df_res.index[0])

    else:
        summary_df = summary_df.truncate(before=df_res.index[0])
        df_res = df_res.truncate(before=summary_df.index[0])

    if df_res.index[-1] > summary_df.index[-1]:
        df_res = df_res.truncate(after=summary_df.index[-1])
        summary_df = summary_df.truncate(after=df_res.index[-1])

    else:
        summary_df = summary_df.truncate(after=df_res.index[-1])
        df_res = df_res.truncate(after=summary_df.index[-1])

    passport_power_table = create_power_from_production_table_py(esp, Freq_, PVTdic['muob_cP'])


    aPathPassport = os.path.join(cWorkFolder,
                                    r'Данные для виртуальной расходометрии\БД', field_name, well_name)
    if not os.path.exists(aPathPassport):
        os.makedirs(aPathPassport)
    plot_pump(passport_power_table, well_name, Freq_start_, PVTdic['muob_cP'], aPathPassport)

    First_liq_point_ = df_res['Q mix pump cond'][0]  # Доверяем первому замеру, берем его за стартовую точку

    expected_q_8h = get_Q_prediction(True, df_res, First_liq_point_, esp, Freq_start_, PVTdic,
                                     passport_power_table, Fw_mean_, well_name, aPathPassport)

    expected_q_1h = get_Q_prediction(False, summary_df, First_liq_point_, esp, Freq_start_, PVTdic,
                                     passport_power_table, Fw_mean_, well_name, aPathPassport)

    summary_df['Expected production'] = expected_q_1h[0]

    summary_df['Expected production on the surface'] = expected_q_1h[4]

    # summary_df['mu emulsion'] = expected_q_1h[2]
    summary_df['Max allowed dQ'] = expected_q_1h[3]
    summary_df['Real dQ'] = summary_df['Expected production'].diff()

    df_res['Expected production daily'] = expected_q_8h[0]

    df_res['Expected daily production on the surface'] = expected_q_8h[4]

    # df_res['mu emulsion'] = expected_q_8h[2]
    df_res['Max allowed dQ'] = expected_q_8h[3]
    df_res['Real dQ'] = df_res['Expected production daily'].diff()

    Mean_error_surface_py = (
            df_res['LIQ_RATE'] - df_res['Expected daily production on the surface']).mean()
    df_res.loc[(df_res['Expected daily production on the surface'] > 0,
                'Expected daily production on the surface')] += Mean_error_surface_py

    summary_df.loc[(summary_df['Expected production on the surface'] > 0,
                    'Expected production on the surface')] += Mean_error_surface_py

    # ДФ для валидации работы алгоритма (выкинут каждый N-й день)
    # Ресемпл с выкинутыми данными по их ключу-индексу

    # Пропуск

    summary_df.to_csv(os.path.join(aPathPassport, 'summary 1H.csv'))
    # validation_df.to_csv(os.path.join(aPath_py, well_name + ' validation.csv'))
    df_res.to_csv(os.path.join(aPathPassport, 'summary 8h.csv'))


    # log_df_py.to_csv(os.path.join(aPathPassport, 'log.csv'), encoding='windows 1251')

    # Выгрузка в Excel

    summary_df_xls = pd.concat([summary_df, qliq_1H], axis=1).reindex()
    reports(summary_df_xls)

    # df_vsp_py = []
    # visualisation(df_res, summary_df, df_vsp_py, well_name)  # df_ml_3h, well_name)

    # print(statistics.variance(df_res['Expected daily production on the surface'] - df_res['Дебит жидкости (ТМ)']))

    print('Calculations data begins at {}, {}, ends at {}, {}, длительность {}'.format(summary_df.index[0], df_res.index[0],
                                                                                       summary_df.index[-1], df_res.index[-1], datetime.now() - calc_time_start))

    print(f'bd to save...{well_name} - {field_name} ({well_id})')

    # df2bd = summary_df.copy()

    df2bd = pd.DataFrame(columns=['WELL_ID', 'PARAM_ID', 'DT', 'VAL'], index=summary_df.index)

    df2bd['DT'] = summary_df.index
    df2bd['VAL'] = summary_df['Expected production on the surface']
    df2bd['WELL_ID'].fillna(value=well_id, inplace=True)  # Подменить пустышки
    df2bd['PARAM_ID'].fillna(value=param_id, inplace=True)  # Подменить пустышки

    config.load_conf('ora_conn_era_neuro')

    sql_delete = f'''
        delete from VIRTUAL_METER_HOUR where well_id = :well_id and param_id = :param_id and
            dt >= :calc_start and dt < :calc_end
        '''
    params = {}
    params['well_id'] = well_id
    params['param_id'] = param_id
    params['calc_start'] = calc_start
    params['calc_end'] = calc_end

    db_wr = writer_bd_vr.writer_bd_base(config, None)
    db_wr.transact_begin()
    db_wr.sql_statement(sql_delete, params)
    db_wr.write_data_df(df2bd, 'VIRTUAL_METER_HOUR')
    db_wr.transact_end()
    db_wr.disconnect()

    print(f'bd to save end... {well_name} - {field_name} ({well_id})')
    pass
