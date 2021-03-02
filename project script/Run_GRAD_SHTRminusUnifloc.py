#!/usr/bin/env python
# coding: utf-8

# In[1]:

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
#get_ipython().run_line_magic('pylab', 'inline')
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:

# Настройка папок
# Рабочей папкой считаем папку выше \Скрипт
cCurrentPath_ = os.path.split(os.path.abspath(''))[0]
print('cCurrentPath_ =', cCurrentPath_)
# cWorkFolder = r'F:\Work'
cWorkFolder = cCurrentPath_
print(cWorkFolder)

# Имена файлов
# Справочник насосов, разделитель табуляция
esp_file_name = r'Данные для виртуальной расходометрии\esp.txt'

# Выбор источника низкочастотных данных

# In[3]:


# если grad = False, то данные подгружаются из ШТР
grad = True
BSI = False #True # Используем высокочастотные сырые данные из БСИ


# Список месторождений

# In[4]:


oilfields = ['Восточно-Пякутинское']
#, 'Суторминское', 'Вынгаяхинское', 'Восточно-Пякутинское'


# Данные по скважине

# In[5]:


Dintake_ = 100 # Диаметр приемной сетки насоса (пока что одинаковый для всех)
KsepGasSep_ = 0.7 # Коэффициент сепарации
#TKsep_ = 89 # Температура сепарации
#Tintake_ = 20 # Температура на приеме
PVT_corr_ = 0 # PVT корреляция для записи в строку PVT

PVTdic = {}
PVTdic['PVTcorr'] = PVT_corr_
PVTdic['ksep_fr'] = float(KsepGasSep_)
PVTdic['qgas_free_sm3day'] = 0

# In[6]:


def get_tube_and_pump_characteristics(well_name, oilfield_):
    aPath = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера\ГРАД')

    df_pump = pd.read_excel(os.path.join(aPath,
        'БОМД.Оборудование-ННГ.xlsx'),
                            sheet_name='Основной_лист', header=3).dropna(axis=1, how = 'all')
    Nominal_production_ = df_pump.loc[(df_pump['№ скважины'] == well_name) & (
                    df_pump['Месторождение'] == oilfield_)]['Подача комплекта, м3'].values[0]
    df_pump = df_pump[['№ скважины', 'Месторождение', 'Глубина спуска','0_Типоразмер_Секции_ЭЦН', '1_Типоразмер_Секции_ЭЦН',
                  '1_Количество_ступеней', '2_Количество_ступеней', '3_Количество_ступеней',
                  '4_Количество_ступеней', '5_Количество_ступеней', '6_Количество_ступеней', '7_Количество_ступеней',
                    '8_Количество_ступеней', '9_Количество_ступеней']]
    df_pump = df_pump.loc[(df_pump['№ скважины'] == well_name) & (df_pump['Месторождение'] == oilfield_)].dropna(axis=1)
    Hpump_ = int(df_pump['Глубина спуска'])
    NumStage_ = int(np.sum(df_pump.iloc[0][4:]))
    ESP_name_ = str(df_pump['1_Типоразмер_Секции_ЭЦН'].values[0])

    # Получаем ID насоса по имени, если имени нет, нужно брать по напору

    df_ESP_id = pd.read_csv(os.path.join(aPath, 'ESP_base_unifloc.csv'), encoding="windows 1251", header=0, sep=';')
    try:
        df_esp = df_ESP_id.loc[df_ESP_id['Модель'] == ESP_name_]
        ESP_id_ = df_esp['ID'].iloc[0]
    except:
        print('Насос не найден, здесь был поиск в БД Unifloc *************************************************  ')

    #df_tube = pd.read_excel('ТР все скв Мессояха (есть НКТ).xlsx', header=2).dropna(axis=1, how='all')
    #df_tube = df_tube[['Скважина', 'Диаметр экспл.колонны', 'Диаметр НКТ']].dropna()
    #try:
    #    Dcas_ = df_tube.loc[df_tube['Скважина'] == well_name+'G']['Диаметр экспл.колонны'].iloc[0]
    #    Dtub_ = df_tube.loc[df_tube['Скважина'] == well_name+'G']['Диаметр НКТ'].iloc[0]
    #except:
    #    Dcas_ = df_tube.loc[df_tube['Скважина'] == well_name+'G2']['Диаметр экспл.колонны'].iloc[0]
    #    Dtub_ = df_tube.loc[df_tube['Скважина'] == well_name+'G2']['Диаметр НКТ'].iloc[0]
    #
    #

    Dcas_ = 178 #Типичный размер ЭК для данного мр-я

    # Значения диаметра НКТ от глубины
    if BSI == False:
        aPath = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера',
                'ГРАД', oilfield_, r'Ствол\НКТ')
        for file in os.listdir(aPath):
            if well_name in file:
                filename = file
        d_tube_df = pd.read_excel(os.path.join(aPath, filename), header=1)
        d_tube_df['Глубина окончания секции'] = d_tube_df['длина,м'][0]
        for i in range(0, len(d_tube_df)):
            d_tube_df['Глубина окончания секции'].loc[i] = np.sum(d_tube_df['длина,м'][:i+1])
        d_tube_df['диаметр(мм)/сортамент'] = d_tube_df['диаметр(мм)/сортамент'].str.slice(0,2)
        d_tube_df['диаметр(мм)/сортамент'].astype(int)
    else:
        d_tube_df = pd.DataFrame()

    data = {'Hpump_' : Hpump_, 'NumStage_' : NumStage_, 'Dcas_' : Dcas_, 'Dtub_' : d_tube_df, 'ESP_id_' : ESP_id_,
           'ESP_name_' : ESP_name_}

    return data


# In[7]:


def get_inclinometry(well_name):

    aPath = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера',
                 'ГРАД', oilfield_, r'Ствол\Инклинометрия')
    for file in os.listdir(aPath):
        if well_name in file:
            filename = file
    df = pd.read_excel(os.path.join(aPath, filename), header=0).dropna(axis=1, how = 'all')
    df['Inc'].astype(float)
    df[['Inc']] = 90 - df[['Inc']]
    return df


# In[8]:


def get_averaged_data(low, well_name):
# Функция считывает данные за 3 часа или 8 часов в зависимости от флага low.
    aPath = ''
    if low:
        if grad and BSI:
            aPath = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера',
                                  'ГРАД', oilfield_, r'Осреднение\High freq')
        else:
            aPath = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск',
                         r'информация для виртуального расходомера\ШТР', oilfield_)

        for file in os.listdir(aPath):
            if well_name in file:
                filename = os.path.join(aPath, file)
        res = get_data_both(low, well_name, filename)

    else:
        if BSI == False:
            aPath = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера',
                                  'ГРАД', oilfield_, r'Осреднение\High freq')
        elif grad and BSI:
            aPath = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера',
                         r'БСИ', oilfield_)
        for file in os.listdir(aPath):
            if well_name in file:
                filename = os.path.join(aPath, file)
        res = get_data_both(low, well_name, filename)

    return res



# In[9]:


def get_data_both(low, well_name, filename):
# Считывание осредненных данных (3 часа или сутки)

    #if low and grad:
    if low and grad:
        df = pd.read_csv(filename, encoding="utf-8-sig", sep=',', index_col=0)
        df.index = pd.to_datetime(df.index)

        #df['Дебит газа (ТМ)'].interpolate(limit_direction='both', inplace=True)
        df['Газовый фактор (рассчитанный)'] = df['Дебит газа (ТМ)'] / df['Дебит нефти (ТМ)']
        df['Газожидкостной фактор (рассчитанный)'] = df['Дебит газа (ТМ)'] / df['Дебит жидкости (ТМ)']
        df['Обводненность (ТМ)'].interpolate(limit_direction='both', inplace=True)

        return df

    elif low == False and grad and BSI:
        df = pd.read_excel(filename)
        df.set_index('Дата, Время', inplace=True)
        df.dropna(subset = ['акт.P,кВт'], inplace=True)

        df.rename(columns={'F, Гц' : 'Частота вращения (ТМ)', 'Tдвиг, °C' : 'Температура двигателя ЭЦН (ТМ)',
                           'акт.P,кВт' : 'Активная мощность (ТМ)', 'P, ат.' : 'Давление на входе ЭЦН (ТМ)'},  inplace=True)
        return df
    else:
        if low:

            df = pd.read_excel(filename, header=2, parse_dates=['Unnamed: 0'], date_parser=dateparser.parse)
            df.rename(columns={'Unnamed: 0' : 'Дата', 'ГФР(ТМ)': 'Газожидкостной фактор (рассчитанный)',
                              'Обв ТМ' : 'Обводненность (ТМ)', 'Рэцн ТМ' : 'Давление на входе ЭЦН (ТМ)',
                              'Тдвиг ТМ' : 'Температура двигателя ЭЦН (ТМ)'}, inplace=True)
            df['Обводненность (ТМ)'] = (1 - df['Qн*'] / df['Qж ТМ']) * 100 # перевод в %
            df['Обводненность (ТМ)'].interpolate(limit_direction='both', inplace=True)
            df.set_index('Дата', inplace=True)
            df = df[['Газожидкостной фактор (рассчитанный)', 'Обводненность (ТМ)', 'Давление на входе ЭЦН (ТМ)',
                    'Температура двигателя ЭЦН (ТМ)']]

            os.chdir(r'F:\Work\Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера'
                r'\ГРАД' + '\\' + oilfield_ + r'\Осреднение\High freq')

            for file in os.listdir(os.getcwd()):
                if well_name in file:
                    filename = file

            df_grad = pd.read_csv(filename, encoding="utf-8-sig", sep=',', index_col=0)
            df_grad.index = pd.to_datetime(df_grad.index)

            if well_name == '6013':
                df_grad.drop(columns = ['Давление на входе ЭЦН (ТМ)', 'Дебит газа (ТМ)',
                                           'Дебит нефти (ТМ)', 'Обводненность (ТМ)'], inplace=True)
            if well_name == '3026':
                df_grad.drop(columns = ['Дебит нефти (ТМ)', 'Обводненность (ТМ)'], inplace=True)


            if 'Давление на входе ЭЦН (ТМ)' in df_grad.columns :
                df = df[['Газожидкостной фактор (рассчитанный)', 'Обводненность (ТМ)']]

            df = pd.concat([df_grad, df], axis=1).reindex(df_grad.index)
            df.dropna(how='all', inplace=True)
        elif grad == True:

            df = pd.read_csv(filename, encoding="utf-8-sig", sep=',', index_col=0)
            df.index = pd.to_datetime(df.index)

            df['Газовый фактор (рассчитанный)'] = df['Дебит газа (ТМ)'] / df['Дебит нефти (ТМ)']
            df['Газожидкостной фактор (рассчитанный)'] = df['Дебит газа (ТМ)'] / df['Дебит жидкости (ТМ)']
            df['Обводненность (ТМ)'].interpolate(limit_direction='both', inplace=True)
        else:
            df = pd.read_csv(filename, encoding="utf-8-sig", sep=',', index_col=0)
            df.index = pd.to_datetime(df.index)

        return df


# In[10]:


def get_data_for_ml(df_calc, df_data):
    # возвращает подготовленные данные
    df = pd.concat([df_calc, df_data.drop(columns=['Давление на входе ЭЦН (ТМ)', 'Температура двигателя ЭЦН (ТМ)',
                                                       'Давление линейное (ТМ)'])], axis=1).reindex()
    return  df


# In[11]:


def get_vsp(oilfield, well_name):
    # Чтение данных о ВСП (внутрисменные простои) скважин
    aPath = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера\ВСП')
    for file in os.listdir(aPath):
        if oilfield in file:
            filename = file
    df = pd.read_csv(filename, encoding = 'windows 1251', sep=';')
    df.drop(columns=['Скв_оис', 'Состояние', 'Источник данных'], inplace=True)
    df.dropna(inplace=True)

    items = []
    for item in df['Скв'].values:
        if '_' in str(item):
            item = int(item[:-2])
        items.append(item)
    df['Скв'] = items
    df = df.loc[df['Скв'] == int(well_name)]
    df['ДатаСтарта'] = pd.to_datetime(df['ДатаСтарта'])
    df['Дата_Окончания'] = pd.to_datetime(df['Дата_Окончания'])
    return df


# In[12]:


def get_data_for_validation(N, df_res, active_power_1H, f_esp_1H, *args):
 # Производим расчеты с выбросом некоторых дней с целью контроля ошибки!


    df_res['K degradation'].iloc[-1] = df_res['K degradation'].iloc[-2]

    each_N_day = df_res.iloc[::N] # Данные за каждый N-й день
    if N > 1:
        #ДФ с днями, на которых проверяем работу алгоритма (выкидывали каждый N-й день)
        validation_days = df_res.drop(index=each_N_day.index)
    else:
    #Для проверки последнего замера
        validation_days = df_res.iloc[-1:]
        each_N_day.drop(index=validation_days.index, inplace=True)

    #resampled_k_deg = each_N_day[['K degradation']].resample('1H')


    if grad and BSI == False:
        resampled_k_deg = df_res[['K degradation']].resample('1H')
        interpolated_k_deg_1H = resampled_k_deg.interpolate('index')
        tKsep_1H = args[0]
        wc_1h = args[1]
        rp_1H = args[2]
        pksep_atma_1H = args[3]
        p_lin_1H = args[4]
        #resampled_f_esp = each_N_day[['F ESP']].resample('1H').interpolate()# Чтобы частота менялась плавнее внутри суток
        #resampled_temp = df_res['Temperature on the surface'].resample('1H').interpolate()

        # На случай, если нет высокочастотной обводненности
        #resampled_fw = each_N_day['Fw'].resample('1H').fillna(method='ffill') #Интерполируем обводненность внутри суток

        #resampled_tempKsep = df_res['Температура двигателя ЭЦН (ТМ)'].resample('1H').interpolate()
        #resampled_p_lin = df_res['Давление линейное (ТМ)'].resample('1H').interpolate()
        interpolated_rp = rp_1H.interpolate()

        return (pd.concat([active_power_1H, f_esp_1H, interpolated_k_deg_1H, tKsep_1H, wc_1h, p_lin_1H, interpolated_rp,
                           pksep_atma_1H], axis=1).reindex(), validation_days)
    elif grad and BSI:
        resampled_k_deg = df_res[['K degradation']].resample('60S')
        interpolated_k_deg_1H = resampled_k_deg.interpolate('index')
        tKsep_1H = args[0]
        pksep_atma_1H = args[1]

        return (pd.concat([active_power_1H, f_esp_1H, interpolated_k_deg_1H, tKsep_1H,
                           pksep_atma_1H], axis=1).reindex(), validation_days)
    else:
        p_lin_1H = args[0]
        resampled_k_deg = df_res[['K degradation']].resample('1H')
        interpolated_k_deg_1H = resampled_k_deg.interpolate('index')
        resampled_tempKsep = df_res['Температура двигателя ЭЦН (ТМ)'].resample('1H').interpolate()
        resmpled_pksep_atma = df_res['Давление на входе ЭЦН (ТМ)'].resample('1H').interpolate()
        resampled_fw = df_res['Fw'].resample('1H').fillna(method='ffill')
        resampled_rp = df_res['Газожидкостной фактор (рассчитанный)'].resample('1H').interpolate()
        return (pd.concat([active_power_1H, f_esp_1H, interpolated_k_deg_1H, resampled_tempKsep, resampled_fw, p_lin_1H,
                           resampled_rp, resmpled_pksep_atma], axis=1).reindex(), validation_days)


# In[13]:


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
        power_w.append(esp.esp_power_w(aqliq_m3day=production[-1])/1000)
        i += 1
    if len(power_w) > 2:
        del power_w[-1]
        del production[-1]

    diff = np.gradient(power_w, production)
    df = pd.DataFrame(columns = ['Production', 'power', 'Derivative'])
    df['Production'] = production
    df['power'] = power_w
    df['Derivative'] = diff
    return df


# In[14]:


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
            N_passport = passport_power_table['power'].values
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


# In[15]:


def get_error_for_optimization(Q, args):

    i = args[0] # Текущая точка в предсказанном массиве
    df = args[1] # ДФ со всеми данными

    Q_true = args[2] #Реальный дебит модели в условиях насоса
    PVTdic = args[3]
    Fw_ = df['Fw'].iloc[i]
    #Temperature = df['Temperature on the surface'].iloc[i]
    PVTdic['pksep_atma'] = df['Давление на входе ЭЦН (ТМ)'].iloc[i]
    PVTdic['tksep_C'] = df['Температура двигателя ЭЦН (ТМ)'].iloc[i]
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


# In[16]:


def transform_to_surface_conditions(Q_pump, PVTdic, i, df):
    Q_start = 100  # Стартовая точка оптимизации

    minimized_result = minimize(get_error_for_optimization, Q_start, method='Nelder-Mead',
                                args=[i, df, Q_pump, PVTdic])  # Запуск оптимизатора
    print('True value is {}, predicted is {}, error is {}'.format(Q_pump, minimized_result.x[0],
                                                                  np.abs(Q_pump - minimized_result.x[0])/Q_pump))
    # print(minimized_result)
    print('{}% completed'.format(np.round(i / len(df) * 100, 2)))

    return minimized_result.x[0]


# In[17]:


def ml_calculations(df_day, df_3h, well_name):

# Расчеты по МО, обучение и контроль ошибки на дневных данных, применяем обученные модели для 3 часовых

    y = df_day['K degradation'].values
    df_day.drop(columns=['K degradation'], inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df_day, y, test_size=0.2, shuffle=True)
    ten_pow = np.power(10.0, np.arange(-4, 2))
    grid = {'alpha': ten_pow}
    cv = KFold(n_splits = 5, shuffle = True)
    clf_test = Ridge()
    gs = GridSearchCV(clf_test, grid, cv=cv)
    gs.fit(X_train,  y_train)
    print(gs.best_estimator_)

    model = gs.best_estimator_
    model.fit(X_train, y_train)


    y_pred_ridge_train = model.predict(X_train)
    y_pred_ridge_test = model.predict(X_test)


    train_r2_ridge = r2_score(y_train, y_pred_ridge_train)
    test_r2_ridge = r2_score(y_test, y_pred_ridge_test)

    train_mae_ridge = mean_absolute_error(y_train, y_pred_ridge_train)
    test_mae_ridge = mean_absolute_error(y_test, y_pred_ridge_test)

    y_res_ridge = model.predict(df_3h)


    model = RandomForestRegressor(max_depth=4, n_estimators=25, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_forest_train = model.predict(X_train)
    y_pred_forest_test = model.predict(X_test)

    train_r2_forest = r2_score(y_train, y_pred_forest_train)
    test_r2_forest = r2_score(y_test, y_pred_forest_test)

    train_mae_forest = mean_absolute_error(y_train, y_pred_forest_train)
    test_mae_forest = mean_absolute_error(y_test, y_pred_forest_test)

    y_res_forest = model.predict(df_3h)

    aPath = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера',
                          'ГРАД', oilfield_, r'Результаты\ML')

    n_features = df_day.shape[1]
    plt.figure(figsize = (25,12))
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), df_day.columns)
    plt.xlabel("Важность признака")
    plt.ylabel("Признак")
    plt.title('Скважина ' + well_name + ' Feature importances')
    plt.tight_layout()
    plt.savefig(os.path.join(aPath, 'Скважина ' + well_name + ' Feature importances' + '.jpg'))
    plt.close();

    return {'Ridge' : y_res_ridge, 'Forest' : y_res_forest , 'train_r2_ridge' : train_r2_ridge ,
            'test_r2_ridge' : test_r2_ridge, 'train_mae_ridge' : train_mae_ridge, 'test_mae_ridge' : test_mae_ridge,
            'train_r2_forest' : train_r2_forest, 'test_r2_forest' : test_r2_forest,
            'train_mae_forest' : train_mae_forest, 'test_mae_forest' : test_mae_forest}


# In[18]:


def visualisation(df_res, summary_df, df_vsp, well_name):#, df_ml_high , well_name):
    aPath = ''
    if grad:
        aPath = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера',
                              'ГРАД', oilfield_, r'Результаты\Plots')
    else:
        aPath = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера',
                              'ГРАД', oilfield_, r'Результаты\Plots\SHTR')

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=summary_df.index,y=summary_df['Expected production'], mode='lines+markers',
                   name="Expected production high freq (pump conditions)")
    )

    fig.add_trace(
        go.Scatter(x=summary_df.index,y=summary_df['Expected production on the surface'], mode='lines+markers',
                   name="Expected production high freq (surface conditions)")
    )

    fig.add_trace(
        go.Scatter(x=df_res.index, y=df_res['Expected production daily'], mode='lines+markers',
                   name="Expected production daily (pump conditions)")
    )

    fig.add_trace(
        go.Scatter(x=df_res.index, y=df_res['Expected daily production on the surface'], mode='lines+markers',
                   name="Expected production daily (surface conditions)")
    )


    fig.add_trace(
        go.Scatter(x=df_res.index, y=df_res['Q mix pump cond'], marker={'size':10}, mode='lines+markers',
                   name="Real production (pump conditions)")
    )

    fig.add_trace(
    go.Scatter(x=df_res.index, y=df_res['Дебит жидкости (ТМ)'], marker={'size':10}, mode='lines+markers',
                   name="Real production (surface)")
    )




    #fig.add_trace(
    #    go.Scatter(x=df_ml_high.index, y=df_ml_high['K ridge'], mode='lines+markers', name="K degradation predicted ridge")
    #)
    #
    #
    #fig.add_trace(
    #    go.Scatter(x=df_ml_high.index, y=df_ml_high['K forest'], mode='lines+markers',
    #               name="K degradation predicted forest")
    #)



    fig.add_trace(
        go.Scatter(x=summary_df.index,y=summary_df['K degradation'], mode='markers', name="K deg interpolated")
    )


    fig.add_trace(
        go.Scatter(x=df_res.index, y=df_res['K degradation'], mode='markers', name="K deg")
    )

    fig.add_trace(
        go.Scatter(x=df_res.index, y=df_res['Газожидкостной фактор (рассчитанный)'], mode='lines+markers',
                   name="Gas-liquid factor")
    )

    #fig.add_trace(
    #    go.Scatter(x=summary_df.index,y=summary_df['Fw'], mode='markers', name="WC")
    #)

    fig.add_trace(
        go.Scatter(x=summary_df.index, y=summary_df['Active power 1H'], mode='lines+markers', name="Active power 1H")
    )

    fig.add_trace(
        go.Scatter(x=summary_df.index, y=summary_df['Expected power'], mode='lines+markers', name="Expected power 1H")
    )

    fig.add_trace(
        go.Scatter(x=summary_df.index, y=summary_df['F ESP'], mode='lines+markers', name="ESP frequency")
    )

    fig.add_trace(
        go.Scatter(x=df_res.index, y=df_res['Active power'], mode='lines+markers', name="Active power (low)")
    )


    fig.add_trace(
        go.Scatter(x=df_res.index, y=df_res['F ESP'], mode='lines+markers', name="ESP frequency (low)")
    )

    fig.add_trace(
        go.Scatter(x=df_res.index, y=df_res['Давление на входе ЭЦН (ТМ)'], mode='lines+markers', name="Intake pressure")
    )

    fig.add_trace(
        go.Scatter(x=df_res.index, y=df_res['Давление линейное (ТМ)'], mode='lines+markers', name="Wellhead pressure")
    )

    #fig.add_trace(
    #    go.Scatter(x=summary_df.index, y=summary_df['Давление линейное (ТМ)'], mode='lines+markers', name="Wellhead pressure")
    #)


    #fig.add_trace(
    #    go.Scatter(x=summary_df.index, y=summary_df['mu emulsion'], mode='lines+markers', name="mu emulsion")
    #)
    #
    #fig.add_trace(
    #    go.Scatter(x=df_res.index, y=df_res['mu emulsion'], mode='lines+markers', name="mu emulsion daily")
    #)
    #
    #fig.add_trace(
    #    go.Scatter(x=df_ml_3h.index, y=df_ml_3h['Давление на входе ЭЦН (ТМ)'],
    #                   mode='lines+markers', name="Pump intake pressure")
    #    )

    if len(df_vsp) != 0:
        shapes=[
            dict(
                type="rect",
                # x-reference is assigned to the x-values
                xref="x",
                # y-reference is assigned to the plot paper [0,1]
                yref="paper",
                x0=df_vsp['ДатаСтарта'].iloc[i],
                y0=0,
                x1=df_vsp['Дата_Окончания'].iloc[i],
                y1=max(summary_df['Expected production']),
                fillcolor="LightSalmon",
                opacity=0.3,
                layer="below",
                line_width=0) for i in range(len(df_vsp))]

        fig.update_layout(
            shapes=shapes
            )
    if grad:
        fig.update_layout(
            title_text= well_name + ' GRAD', template='plotly_white'
        )
    else:
        fig.update_layout(
            title_text= well_name + ' SHTR', template='plotly_white'
        )


    fig.update_xaxes(title_text="Дата")

    #fig.update_yaxes(title_text="Production m3/day")
    if grad:
        aPathWithFileName = os.path.join(aPath, well_name + ' GRAD.html')
        plotly.offline.plot(fig, filename = aPathWithFileName, auto_open=False)
    else:
        aPathWithFileName = os.path.join(aPath, well_name + ' SHTR.html')
        plotly.offline.plot(fig, filename = aPathWithFileName, auto_open=False)

    #fig.show()


# In[19]:


def plot_pump(passport_power_table, well_name, Freq_, mu_, aPath):
    # Функция построения напорно-расходной хар-ки по мощности
    fig,ax = plt.subplots(figsize=(15,15))
    plt.tight_layout()
    ax.plot(passport_power_table['Production'], passport_power_table['power'])
    ax.set_xlabel("Production",fontsize=14)
    ax.set_ylabel("power",fontsize=14)
    ax2=ax.twinx()
    ax2.plot(passport_power_table['Production'], passport_power_table['Derivative'], color='red')
    ax2.set_ylabel("Derivative",fontsize=14)
    aFileName = well_name + ' Freq = ' + str(Freq_) + ' mu = ' + str(mu_) + '.jpg'
    plt.savefig(os.path.join(aPath, aFileName), quality=100)
    plt.close()


# In[20]:


def plot_error(validation_df, well_name):
        # Функция постороения напорно-расходной хар-ки по мощности
    path = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера',
                          'ГРАД', oilfield_, r'Результаты\Error', well_name + ' cross-plot')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.figure(figsize = (10,10))
    plt.title(path)
    plt.scatter(validation_df['Q mix pump cond'], validation_df['Expected production'])
    plt.xlabel('Q mix pump cond')
    plt.ylabel('Expected production')
    plt.savefig(os.path.join(path, well_name + ' cross-plot' + '.jpg'), quality=100)
    plt.close()


# In[21]:


def test_optimization(df):
    Q_for_test = []
    for i in range(len(df)):


        Q = df['Expected daily production on the surface'].iloc[i]
        Fw_ = df['Fw'].iloc[i]
        Pksep_atma_ = df['Давление на входе ЭЦН (ТМ)'].iloc[i]
        TKsep_ = df['Температура двигателя ЭЦН (ТМ)'].iloc[i]
        Rp_ = df['Газожидкостной фактор (рассчитанный)'].iloc[i]
        PVT_str = PVT_encode_string(gamma_gas_, gamma_oil_,gamma_wat_, Rsb_, Rp_, Pb_, Tres_, Bob_, mu_, PVT_corr_,
                                    KsepGasSep_, float(Pksep_atma_), float(TKsep_))

        Q_for_test.append(MF_q_mix_rc_m3day(float(Q), Fw_, float(Pksep_atma_), float(TKsep_), PVT_str))

    df['Q test pump conditions'] = Q_for_test

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df.index,y=df['Q test pump conditions'], mode='lines+markers',
                   name="Test pump")
    )

    fig.add_trace(
        go.Scatter(x=df.index,y=df['Expected production daily'], mode='lines+markers',
                   name="Expected production high freq (pump conditions)")
    )

    fig.update_layout(
        title_text='Test', template='plotly_white'
    )
    fig.update_xaxes(title_text="Дата")


    plotly.offline.plot(fig, filename = 'Optimization test.html', auto_open=False)

def prepare_esp_base(file_name):
    curr_path = os.path.join(cWorkFolder, file_name)
    esp_table = pd.read_table(curr_path)
    return esp_table

def reports(summary_df_xls, field_name, well_name, path):
    summary_df_xls.rename(columns={'Expected power': 'Мощность насоса ожид',
                                  'Expected production': 'Qж (ТМ) в усл.насоса ожид',
                                  'Expected production on the surface': 'Qж (ТМ) в поверх.услов. ожид',
                                   },
                         inplace=True)

    summary_df_xls.to_excel(os.path.join(path, 'summary 1H.xls'))

    # Построение графиков

    # Сравнение входного Qж (ТМ) и расчетного в поверхностных условиях
    fig, ax = plt.subplots(figsize=(20, 20))
    plt.tight_layout()

    ax.scatter(summary_df_xls.index, summary_df_xls['Дебит жидкости (ТМ)'], color='blue', label = 'Qж (ТМ)')
    ax.plot(summary_df_xls.index, summary_df_xls['Qж (ТМ) в поверх.услов. ожид'], color='green', label = 'Qж (ТМ) в поверх.услов. ожид')

    ax.legend()
    ax.set_title(f'{field_name} {well_name} Сравнение входного Qж (ТМ) и расчетного в поверхностных условиях ({summary_df_xls.index[0]} - {summary_df_xls.index[-1]})')

    aFileName = field_name + ' ' + well_name + '_qliq_expected_vs_tm_minusU.jpg'
    plt.savefig(os.path.join(path, aFileName), quality=150)
    plt.close()

    # fig, ax = plt.subplots(figsize=(15, 15))
    # plt.tight_layout()
    #
    # ax.scatter(summary_df_xls.index, summary_df_xls['Активная мощность (ТМ)'], color='blue', label = 'Активная мощность (ТМ)')
    # ax.plot(summary_df_xls.index, summary_df_xls['Мощность насоса ожид'], color='red', label = 'Мощность насоса расчетная')
    #
    # ax.legend()
    # ax.set_title(f'{field_name} {well_name} Сравнение Активной мощности (ТМ) и расчетной ({summary_df_xls.index[0]} - {summary_df_xls.index[-1]})')
    #
    # aFileName = field_name + ' '+ well_name + '_activ_power_expected.jpg'
    # plt.savefig(os.path.join(path, aFileName), quality=100)
    # plt.close()

# Подключение функций Unifloc

# In[22]:


# Вычисления

# In[23]:


for oilfield_ in oilfields:

    if oilfield_ == 'Суторминское':
        gamma_oil_ = 0.808
        gamma_wat_ = 1
        gamma_gas_ = 0.754
        Rsb_ = 15.6
        Rp_ = 186
        Pb_ = 138.7
        Tres_ = 76
        Bob_ = 1.486
        mu_ = 2.846 # Вязкость нефти, из PVT
    elif oilfield_ == 'Вынгаяхинское':
        # Из ОИС
        gamma_oil_ = 0.777
        gamma_wat_ = 1.054
        gamma_gas_ = 0.749
        Rsb_ = 15.6 # 267
        Rp_ = 267
        Pb_ = 139 #243
        Tres_ = 101.5
        Bob_ = 1.582
        mu_ = 5.58

        # Константы Excel
        # gamma_oil_ = 0.82
        # gamma_wat_ = 1
        # gamma_gas_ = 0.796
        # Rsb_ = 15.6
        # Rp_ = 124
        # Pb_ = 139
        # Tres_ = 89
        # Bob_ = 1.278
        # mu_ = 7.76 # Вязкость нефти, из PVT
    elif oilfield_ == 'Восточно-Пякутинское':
        gamma_oil_ = 0.846
        gamma_wat_ = 1
        gamma_gas_ = 0.712
        Rsb_ = 15.6
        Rp_ = 73
        Pb_ = 122.4
        Tres_ = 85
        Bob_ = 1.168
        mu_ = 12.96 # Вязкость нефти, из PVT

    PVTdic['gamma_gas'] = float(gamma_gas_)
    PVTdic['gamma_oil'] = float(gamma_oil_)
    PVTdic['gamma_wat'] = float(gamma_wat_)
    PVTdic['rsb_m3m3'] = float(Rsb_)
    PVTdic['pb_atma'] = float(Pb_)
    PVTdic['tres_C'] = float(Tres_)
    PVTdic['bob_m3m3'] = float(Bob_)
    PVTdic['muob_cP'] = float(mu_)
    PVTdic['bwSC_m3m3'] = PVTdic['gamma_wat'] / 1
    PVTdic['salinity_ppm'] = calc_well_param.ccalc_pvt.unf_calc_Sal_BwSC_ppm(PVTdic['bwSC_m3m3'])


        # Список скважин

    aPath = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера',
                          'ГРАД', oilfield_, 'Скважины')

    filenames = []
    for file in os.listdir(aPath):
        filenames.append(file[:-4])
    print(filenames)

    N = 1 # Параметр для проверки качества модели (выкидываем каждую N-ю точку и считаем ошибку)

    #filenames = ['2355']

################# вызов Виртуальный Расходомер начало
    log_df_py = pd.DataFrame(index=[filenames], columns=['ESP name real', 'ESP name used', 'ESP id',
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
################# вызов Виртуальный Расходомер 

    for names in zip(range(len(filenames)), filenames):

        well_name = names[1]

        if well_name in ['6011', '6013', '1976', '3026', '337', '2338', '2355', '3900', '3922']:
            grad = False
        else:
            grad = True

        tube_and_pump = get_tube_and_pump_characteristics(well_name, oilfield_)

        Hpump_ = tube_and_pump['Hpump_']
        NumStage_ = tube_and_pump['NumStage_']
        Dcas_ = tube_and_pump['Dcas_']
        #Dtub_ = tube_and_pump['Dtub_']
        Dtub_ = 79 #нехорошо
        ESP_id_ = int(tube_and_pump['ESP_id_'])
        ESP_name_ = tube_and_pump['ESP_name_']

################# вызов Виртуальный Расходомер начало
        esp_base = prepare_esp_base(esp_file_name)
        pumps_for_id = esp_base[esp_base['ID'] == ESP_id_]
        ESP_name_unifloc_py = pumps_for_id['Модель'].iloc[0]
        ESP_manufacturer_py = pumps_for_id['Производитель'].iloc[0]
        ESP_freq_py = int(pumps_for_id['Частота'].iloc[0])
        ESP_nom_rate_py = int(pumps_for_id['Номинал'].iloc[0])
        ESP_stage_num_py = int(pumps_for_id['Ступеней макс'].iloc[0])
        ESP_max_rate_m3day_py = pumps_for_id['Дебит'].astype('float').max()

        esp_rates = pumps_for_id['Дебит'].astype('float').to_list()
        esp_heads = pumps_for_id['Напор'].astype('float').tolist()
        esp_powers = pumps_for_id['Мощность'].astype('float').tolist()
        esp_eff = pumps_for_id['КПД'].astype('float').tolist()


        esp_heads_с = calc_well_param.cesp.polinom_solver(esp_rates, esp_heads, 5)
        esp_polynom_head = calc_well_param.cesp.esp_polynom(esp_heads_с)

        esp_eff_с = calc_well_param.cesp.polinom_solver(esp_rates, esp_eff, 5)
        esp_polynom_efficency = calc_well_param.cesp.esp_polynom(esp_eff_с)

        esp_powers_с = calc_well_param.cesp.polinom_solver(esp_rates, esp_powers, 5)
        esp_polynom_power = calc_well_param.cesp.esp_polynom(esp_powers_с)

        ################# вызов Виртуальный Расходомер

        # По ШТР
        if grad == False and BSI == False:
            all_data = get_averaged_data(True, well_name)
            all_data_high = get_averaged_data(False, well_name)

        # По БСИ
        elif grad and BSI:
            all_data_high = get_averaged_data(False, well_name).resample('60S').mean()
            all_data = get_averaged_data(True, well_name)

        elif grad and BSI == False:
            all_data_high = get_averaged_data(False, well_name)
            all_data = all_data_high.dropna(subset=['Дебит жидкости (ТМ)'])
        else:
            print('ГЫГ')
            all_data = get_averaged_data(True, well_name)
            all_data.dropna(subset=['Дебит жидкости (ТМ)'], inplace=True)

        #all_data.dropna(subset=['Дебит жидкости (ТМ)'], inplace=True)
        #all_data['Активная мощность (ТМ)'].interpolate(limit_direction='both', inplace=True)
        all_data['Газожидкостной фактор (рассчитанный)'].interpolate(limit_direction='both', inplace=True)
        all_data['Частота вращения (ТМ)'].interpolate(limit_direction='both', inplace=True)
        all_data['Температура двигателя ЭЦН (ТМ)'].interpolate(limit_direction='both', inplace=True)

        #df_vsp = get_vsp(oilfield_, well_name)

        if all_data.index[0] < all_data_high.index[0]:
            all_data = all_data.truncate(before = all_data_high.index[0])
            all_data_high = all_data_high.truncate(before = all_data.index[0])

        else:
            all_data_high = all_data_high.truncate(before = all_data.index[0])
            all_data = all_data.truncate(before = all_data_high.index[0])

        if all_data.index[-1] > all_data_high.index[-1]:
            all_data = all_data.truncate(after = all_data_high.index[-1])
            all_data_high = all_data_high.truncate(after = all_data.index[-1])

        else:
            all_data_high = all_data_high.truncate(after = all_data.index[-1])
            all_data = all_data.truncate(after = all_data_high.index[-1])

        #if df_vsp.index[0] < all_data.index[0]:
        #    df_vsp = df_vsp.truncate(before = all_data.index[0])
        #if df_vsp.index[-1] > all_data.index[-1]:
        #    df_vsp = df_vsp.truncate(after = all_data.index[-1])


        print('Input data begins at {}, {}, ends at {}, {}'.format(all_data_high.index[0], all_data.index[0],
                                                       all_data_high.index[-1], all_data.index[-1]))
        # Проверка на Мпа/атм
        print('Average pressure, ', all_data['Давление линейное (ТМ)'].dropna().mean())
        if all_data['Давление линейное (ТМ)'].dropna().mean() < 3:
            all_data['Давление линейное (ТМ)'] = all_data['Давление линейное (ТМ)'] * 9.869
            if BSI == False:
                all_data_high['Давление линейное (ТМ)'] = all_data_high['Давление линейное (ТМ)'] * 9.869

        if grad == False and BSI == False:
            active_power_1H = all_data_high[['Активная мощность (ТМ)']]
            f_esp_1H = all_data_high[['Частота вращения (ТМ)']]
            p_lin_1H = all_data_high[['Давление линейное (ТМ)']]

        if grad and BSI == False:
            active_power_1H = all_data_high[['Активная мощность (ТМ)']]
            f_esp_1H = all_data_high[['Частота вращения (ТМ)']]
            p_lin_1H = all_data_high[['Давление линейное (ТМ)']]
            tKsep_1H = all_data_high[['Температура двигателя ЭЦН (ТМ)']]
            pksep_atma_1H = all_data_high[['Давление на входе ЭЦН (ТМ)']]
            rp_1H = all_data_high[['Газожидкостной фактор (рассчитанный)']]
            wc_1h = all_data_high[['Обводненность (ТМ)']]
            qliq_1H = all_data_high[['Дебит жидкости (ТМ)']]

        if grad and BSI:
            active_power_1H = all_data_high[['Активная мощность (ТМ)']]#.resample('60S').mean()
            f_esp_1H = all_data_high[['Частота вращения (ТМ)']]#.resample('60S').mean()
            tKsep_1H = all_data_high[['Температура двигателя ЭЦН (ТМ)']]#.resample('60S').mean()
            pksep_atma_1H = all_data_high[['Давление на входе ЭЦН (ТМ)']]#.resample('60S').mean()



        f_esp_1H.rename(columns = {'Частота вращения (ТМ)' : 'F ESP'}, inplace=True)
        inclinometry = get_inclinometry(well_name)

        ################# вызов Виртуальный Расходомер начало
        pressure_at_the_discharge_py = []

        k_deg_py = [] # Массив суточных коэффициентов деградации

        passp_pow_py = [] # Массив паспортных мощностей раз в сутки по результатам расчета python
        q_mix_array_py = [] # Массив суточных дебитов в условиях насоса по результатам расчета python
        ################# вызов Виртуальный Расходомер

        Freq_start_ = all_data['Частота вращения (ТМ)'].iloc[0] # частота ЭЦН в первый день
        ################# вызов Виртуальный Расходомер начало
        df_res_py = pd.DataFrame(index = all_data.index, columns = ['Q mix pump cond', 'K degradation'])
        ################# вызов Виртуальный Расходомер начало

        for i in range(len(all_data)):

            Pksep_atma_ = float(all_data['Давление на входе ЭЦН (ТМ)'].iloc[i])
            Rp_ = all_data['Газожидкостной фактор (рассчитанный)'].iloc[i]
            Qliq_ = all_data['Дебит жидкости (ТМ)'].iloc[i]
            Fw_ = all_data['Обводненность (ТМ)'].iloc[i]
            Fw_mean_ = all_data[['Обводненность (ТМ)']].mean()#[0]
            Active_pow_ = all_data['Активная мощность (ТМ)'].iloc[i]

            #mu_ = int(include_emulsion(Fw_mean_))
            #mu_ = int(include_emulsion(Fw_))

            Pintake_ = Pksep_atma_
            Plin_ = all_data['Давление линейное (ТМ)'].iloc[i]
            Freq_ = float(all_data['Частота вращения (ТМ)'].iloc[i])
            TKsep_ = all_data['Температура двигателя ЭЦН (ТМ)'].iloc[i]
            Tintake_ = TKsep_


            PVTdic['pksep_atma'] = float(Pksep_atma_)
            PVTdic['tksep_C'] = float(TKsep_)
            PVTdic['rp_m3m3'] = float(Rp_)

            pressure_full_temp_py = [Plin_] # Распределение давления по стволу (свое в каждый день)

            j = 1
            while inclinometry['MD'].iloc[j] < Hpump_:
                Length_ = inclinometry['MD'].iloc[j] - inclinometry['MD'].iloc[j-1]
                Pcalc_ = pressure_full_temp_py[j-1]
                Calc_along_flow_ = 0
                Theta_deg = inclinometry['Inc'].iloc[j]
                Hydr_corr_ = 0
                Tcalc_ = Tres_
                Tother_ = Tcalc_

                ################# вызов Виртуальный Расходомер начало
                if Pksep_atma_ == Pksep_atma_ and Qliq_ == Qliq_ and Fw_ == Fw_:
                    pressure_py = calc_well_param.cpipe.pipe_atma(Qliq_, Fw_, float(Length_), Pcalc_, Calc_along_flow_,
                                                                  PVTdic, Theta_deg, Dtub_, Hydr_corr_, Tcalc_, Tother_)
                    pressure_full_temp_py.append(pressure_py[0])
                else:
                    pressure_full_temp_py.append(0)

                #print(f'    pressure_unifloc = {pressure_unifloc}, pressure_py = {pressure_py}, Length_ = {Length_}, Pcalc_ = {Pcalc_}, Theta_deg = {Theta_deg}')
                ################# вызов Виртуальный Расходомер
                j += 1

################# вызов Виртуальный Расходомер начало
            z_class = calc_well_param.calc_func.z_factor_2015_kareem()
            z = z_class.z_factor
            if Pksep_atma_ == Pksep_atma_ and Qliq_ == Qliq_ and Fw_ == Fw_:
                PVTdic = calc_well_param.ccalc_pvt.calc_pvt_vr(Pintake_, Tintake_, Qliq_, Fw_ / 100, PVT_corr_, PVTdic, z)


#            if Pksep_atma_ == Pksep_atma_ and Qliq_ == Qliq_ and Fw_ == Fw_:
#                PVTdic = calc_well_param.ccalc_pvt.calc_pvt_vr(Pintake_, Tintake_, Qliq_, Fw_ / 100, PVT_corr_, PVTdic,
            #                calc_well_param.calc_func.factor_2015_kareem)

            if 'q_mix_rc_m3day' in PVTdic:
                Q_mix_intake_py = PVTdic['q_mix_rc_m3day']
            else:
                PVTdic.update({'q_mix_rc_m3day': 0})
            pressure_at_the_discharge_py.append(pressure_full_temp_py[-1])
            # print('Unifloc Q_mix_intake_ =', Q_mix_intake_, 'py Q_mix_intake_py =', Q_mix_intake_py)
            print('py pressure_full_temp_py[-1] =', pressure_full_temp_py[-1])

            z_class = calc_well_param.calc_func.z_factor_2015_kareem()
            z = z_class.z_factor
            PVTdic = calc_well_param.ccalc_pvt.calc_pvt_vr(pressure_at_the_discharge_py[-1], Tintake_, Qliq_, Fw_ / 100, PVT_corr_, PVTdic, z)

#            PVTdic = calc_well_param.ccalc_pvt.calc_pvt_vr(pressure_at_the_discharge_py[-1], Tintake_, Qliq_, Fw_ / 100,
            #            PVT_corr_, PVTdic, calc_well_param.calc_func.factor_2015_kareem)
            Q_mix_discharge_py = PVTdic['q_mix_rc_m3day']
            Q_mix_py = (Q_mix_intake_py + Q_mix_discharge_py) / 2 # Дебит ГЖС в условиях насоса
            # print('Unifloc Q_mix_ =', Q_mix_, 'py Q_mix_py =', Q_mix_py)
            q_mix_array_py.append(Q_mix_py)

            new_esp = calc_well_param.cesp.esp(id_pump=0, manufacturer_name=ESP_manufacturer_py,
                                               pump_name=ESP_name_unifloc_py,
                                               freq_hz=ESP_freq_py, esp_nom_rate_m3day=ESP_nom_rate_py,
                                               esp_max_rate_m3day=ESP_max_rate_m3day_py,
                           esp_polynom_head_obj=esp_polynom_head, esp_polynom_efficency_obj=esp_polynom_efficency,
                           esp_polynom_power_obj=esp_polynom_power)

            new_esp.mu_cSt = float(mu_)
            new_esp.correct_visc = True
            new_esp.freq_hz = float(Freq_)
            new_esp.stage_num = NumStage_
            power_w = new_esp.esp_power_w(aqliq_m3day=Q_mix_py)/1000
            k_deg_py.append(Active_pow_/power_w)

            # print('Unifloc Passport_pow_ =', Passport_pow_, 'py power_w =', power_w)
            passp_pow_py.append(power_w)
################# вызов Виртуальный Расходомер

################# вызов Виртуальный Расходомер начало
        df_res_py['Q mix pump cond'] = q_mix_array_py
        df_res_py['K degradation'] = k_deg_py
        df_res_py['F ESP'] = all_data['Частота вращения (ТМ)']
        df_res_py['Expected power'] = passp_pow_py
        df_res_py['Active power'] = all_data['Активная мощность (ТМ)']
        df_res_py['dN expected'] = df_res_py['Expected power'].diff() # Изменение активной мощности в 8 часовых данных
        df_res_py['dN expected'].fillna(method='backfill', inplace=True)
        df_res_py['Газожидкостной фактор (рассчитанный)'] = all_data['Газожидкостной фактор (рассчитанный)']

        df_res_py['Fw'] = all_data['Обводненность (ТМ)']
        df_res_py['Давление на входе ЭЦН (ТМ)'] = all_data['Давление на входе ЭЦН (ТМ)']
        df_res_py['Температура двигателя ЭЦН (ТМ)'] = all_data['Температура двигателя ЭЦН (ТМ)']
        df_res_py['Дебит жидкости (ТМ)'] =  all_data['Дебит жидкости (ТМ)']
        df_res_py['Давление линейное (ТМ)'] = all_data['Давление линейное (ТМ)']

        #Данные за каждые сутки без ошибок в вычислениях
        df_res_py.drop(df_res_py[df_res_py['Q mix pump cond']==-1].index, inplace=True)
        #df_res_py[['F ESP']] = df_res_py[['F ESP']].interpolate() # На случай пропусков в суточных данных
        df_res_py.dropna(inplace=True) # На случай пропусков в суточных данных

        Mean_prod_py = df_res_py['Q mix pump cond'].mean() # Средний дебит по скважине (после выброса -1!)

        # Производим расчеты с выбросом некоторых дней с целью контроля ошибки!
        summary_df_py = get_data_for_validation(N, df_res_py, active_power_1H, f_esp_1H, tKsep_1H, wc_1h,
                                             rp_1H, pksep_atma_1H, p_lin_1H)[0]

        # = pd.concat([active_power_3H, interpolated_k_deg_3H, resampled_f_esp, resampled_fw], axis=1).reindex()

        # Подумать, стоит ли так делать
        # Пропуски мощности инт-ем
        # summary_df['Активная мощность (ТМ)'] = summary_df['Активная мощность (ТМ)'].interpolate()

        summary_df_py.dropna(inplace=True)

# Ожидаемая активная мощность
        summary_df_py['Expected power'] = summary_df_py['Активная мощность (ТМ)'] / summary_df_py['K degradation']
        
        summary_df_py['dN expected'] = summary_df_py['Expected power'].diff()  # Изменение активной мощности
        summary_df_py['dN expected'].fillna(method='backfill', inplace=True)
        
        summary_df_py.rename(columns={'Активная мощность (ТМ)': 'Active power 1H', 'Обводненность (ТМ)': 'Fw'}, inplace=True)


################# вызов Виртуальный Расходомер начало
# Приводим к одному моменту начала py

        if df_res_py.index[0] < summary_df_py.index[0]:
            df_res_py = df_res_py.truncate(before=summary_df_py.index[0])
            summary_df_py = summary_df_py.truncate(before=df_res_py.index[0])
        
        else:
            summary_df_py = summary_df_py.truncate(before=df_res_py.index[0])
            df_res_py = df_res_py.truncate(before=summary_df_py.index[0])
        
        if df_res_py.index[-1] > summary_df_py.index[-1]:
            df_res_py = df_res_py.truncate(after=summary_df_py.index[-1])
            summary_df_py = summary_df_py.truncate(after=df_res_py.index[-1])
        
        else:
            summary_df_py = summary_df_py.truncate(after=df_res_py.index[-1])
            df_res_py = df_res_py.truncate(after=summary_df_py.index[-1])
        
        print('Calculations data begins at {}, {}, ends at {}, {}'.format(summary_df_py.index[0], df_res_py.index[0],
                                                                  summary_df_py.index[-1], df_res_py.index[-1]))

################# вызов Виртуальный Расходомер начало
        esp = calc_well_param.cesp.esp(id_pump=0, manufacturer_name=ESP_manufacturer_py,
                                           pump_name=ESP_name_unifloc_py,
                                           freq_hz=ESP_freq_py, esp_nom_rate_m3day=ESP_nom_rate_py,
                                           esp_max_rate_m3day=ESP_max_rate_m3day_py,
                           esp_polynom_head_obj=esp_polynom_head, esp_polynom_efficency_obj=esp_polynom_efficency,
                           esp_polynom_power_obj=esp_polynom_power)

        esp.stage_num = NumStage_
        passport_power_table_py =create_power_from_production_table_py(esp, Freq_, PVTdic['muob_cP'])

################# вызов Виртуальный Расходомер начало
        print('Виртуальный Расходомер предсказание начала')
        aPathPassport_py = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера',
                          'ГРАД', oilfield_, r'Результаты_py\Pump passport')

        plot_pump(passport_power_table_py, well_name, Freq_start_, mu_, aPathPassport_py)

        First_liq_point_py = df_res_py['Q mix pump cond'][0] # Доверяем первому замеру, берем его за стартовую точку

        expected_q_8h_py = get_Q_prediction(True, df_res_py, First_liq_point_py, esp, Freq_start_, PVTdic,
                                         passport_power_table_py, Fw_mean_, well_name, aPathPassport_py)

        expected_q_1h_py = get_Q_prediction(False, summary_df_py, First_liq_point_py, esp, Freq_start_, PVTdic,
                                         passport_power_table_py, Fw_mean_, well_name, aPathPassport_py)

        summary_df_py['Expected production'] = expected_q_1h_py[0]

        summary_df_py['Expected production on the surface'] =  expected_q_1h_py[4]

        #summary_df_py['mu emulsion'] = expected_q_1h[2]
        summary_df_py['Max allowed dQ'] = expected_q_1h_py[3]
        summary_df_py['Real dQ'] = summary_df_py['Expected production'].diff()


        df_res_py['Expected production daily'] = expected_q_8h_py[0]

        df_res_py['Expected daily production on the surface'] = expected_q_8h_py[4]

        #df_res_py['mu emulsion'] = expected_q_8h[2]
        df_res_py['Max allowed dQ'] = expected_q_8h_py[3]
        df_res_py['Real dQ'] = df_res_py['Expected production daily'].diff()
        print('Виртуальный Расходомер предсказание окончание')
################# вызов Виртуальный Расходомер

################# вызов Виртуальный Расходомер начало
        # Ошибка при восстановлении дебита на поверхности почти константа, делаем смещение расчетов (костыль, колхоз) - py

        Mean_error_surface_py = (df_res_py['Дебит жидкости (ТМ)'] - df_res_py['Expected daily production on the surface']).mean()
        df_res_py.loc[(df_res_py['Expected daily production on the surface'] > 0,
                    'Expected daily production on the surface')] += Mean_error_surface_py

        summary_df_py.loc[(summary_df_py['Expected production on the surface'] > 0,
                        'Expected production on the surface')] += Mean_error_surface_py

        #summary_df_py_ml['Q mix pump cond'] = summary_df_py['Expected production'] #??????????????????????????????????????

################# вызов Виртуальный Расходомер начало
        aPath_py = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера',
                              'ГРАД', oilfield_, r'Результаты_py\Calc results')

        summary_df_py.to_csv(os.path.join(aPath_py, well_name + ' summary 1H.csv'))
        df_res_py.to_csv(os.path.join(aPath_py, well_name + ' summary 8h.csv'))
        aPath_py = os.path.join(cWorkFolder, r'Данные для виртуальной расходометрии\Ноябрьск\информация для виртуального расходомера',
                              'ГРАД', oilfield_, 'Результаты_py')

        #print(statistics.variance(df_res['Expected daily production on the surface'] - df_res['Дебит жидкости (ТМ)']))

        print('** py: well № {} calculated, total progress is {} %'.format(str(names[1]),
            (int(names[0])+1)/len(filenames) * 100))

        summary_df_xls = pd.concat([summary_df_py, qliq_1H], axis=1).reindex()
        reports(summary_df_xls, oilfield_, well_name, aPath_py)

################# вызов Виртуальный Расходомер

# In[ ]:


# N = np.array([25, 15, 14, 12, 11, 10, 11, 12, 14, 15, 20, 25, 30, 35, 37, 38, 39, 40, 39, 35,24])
# Q = range(1, len(N)+1, 1)
# Q_result = np.empty(0)
# Q_for_interpolating = []
# N_for_interpolating = []
# plt.plot(N, Q)
# n_new = 24.8
# vert = [n_new]*len(Q)
# plt.plot(vert,Q)
# Q_point = 10
# for i in range(1, len(N)):
#     if (N[i-1] <= n_new and N[i] >= n_new) or (N[i-1] >= n_new and N[i] <= n_new):
#         print(N[i-1], N[i])
#         N_for_interpolating = [N[i-1], N[i]]
#         Q_for_interpolating = [Q[i-1], Q[i]]
#         Inter = scipy.interpolate.interp1d(N_for_interpolating, Q_for_interpolating)
#         Q_result = np.append(Q_result, Inter(n_new))
#
#         print('Possible Q is', Q_result)
# res = Q_result.flat[np.abs(Q_result - Q_point).argmin()]
# print('Selecting Q= ', res)


print('Расчет завершен')
