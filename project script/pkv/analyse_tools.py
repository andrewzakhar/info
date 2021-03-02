import math
from datetime import datetime

from Скрипт.pkv import tools
import time
import pandas as pd
import numpy as np
from Скрипт.pkv.db_writers import common_writer as cw
from Скрипт.pkv.tools import EspIndex as ei

gn = tools.GlobalNames()

parameters = [gn.work_status_number, "Рабочая доля времени", gn.current_dt, gn.q_liq_m3day, gn.q_oil_mass_tday,
              gn.q_gas_m3day,
              gn.active_power_kwt, gn.i_a_motor_a, gn.motor_load_perc, gn.freq_hz, gn.cos_phi_d,
              gn.p_intake_atm, gn.p_lin_atm, gn.t_intake_c, gn.t_motor_c,
              gn.vibration_xy_msec2, gn.vibration_z_msec2, gn.freq_turb_hz
              ]
default_settings = {
    'days_to_analyse': -1,
    # настройки для определения поступления газа в ЭЦН
    'left_boundary': 0.35,
    'right_boundary': 1,  # лучше не брать последнюю точку если последняя точка нулевая
    'falling_param_val': -0.11,
    'up_down_param_up_val': 0.06,
    'up_down_param_down_val': -0.05,
    'overall_down': -0.3,
    'crit_value_for_load': 45
}


def find_gas_periods(this_df, param=gn.i_a_motor_a, settings=default_settings):
    """
    Функция для поиска поступления газа в ЭЦН при ПКВ.

    Основная идея - обезразмеривание рабочего периода времени (по оси t и y) и поиск падений параметра в конце режима

    param: параметр для поиска поступлений газа
    settings: словарь с настройками

    return very_bad_periods, gas_dfs, stats_with_gas, df
            very_bad_periods - периоды работы, в которых произошла остановки из-за газа
            gas_dfs - список DataFrame с поступлениями газа в ЭЦН
            stats_with_gas - статистика по газу
            df - DataFrame с поступлением газа
    """
    df = this_df.loc[this_df['param_id'] == int(param)].dropna(subset=['val']).sort_values('dt')

    work_status, work_timedelta, stop_timedelta, work_bounds, stop_bounds, stats, _, _ = tools.calculated_regime_time(
        df, regime_parameter=param, return_all=True)
    df[gn.work_status_number] = work_status

    work_periods = []  # создадим маленькие датафреймы для рабочих периодов для безопасной работы с ними
    for k in work_bounds:
        temp = df[(df['dt'] >= k[0]) & (df['dt'] <= k[1])]
        work_periods.append(temp)

    overall_down_list = []
    overall_down_in_percent = []
    time_to_decrease_work_period_sec = []
    amount_of_points_for_each_work_period = []
    amount_of_points_for_each_survey = []
    gas_periods = []
    gas_dfs = []
    very_bad_periods = []
    very_bad_dfs = []
    if len(work_periods) > 2:  # проверим, что периодов для анализа достаточно
        for j, i in enumerate(work_periods):  # пройдемся по каждому рабочему периоду
            small_df = i.copy()
            small_df = small_df.dropna(subset=['val'])
            amount_of_points_for_each_work_period.append(
                small_df.shape[0])  # запишем общее количество точек рабочего периода
            if small_df.shape[0] > 0:
                small_df = small_df.set_index('dt')
                small_df = small_df['val']
                # минимальные значения уберем (особенно важно для тока), по загрузке и так работает
                small_df = small_df[small_df > 3]
                if len(small_df) > 0:
                    small_df = small_df / small_df.max()  # обезразмерим по y
                    small_df_with_init_index = small_df.copy()  # сохраним df с нормальным индексом
                    small_df.index = tools.undim_index(pd.DatetimeIndex(small_df.index))  # обезразмерим по x

                    # сделать проверку по медианному значению
                    # возьмем правую часть интервала, т.к. поступление газа происходят в конце работы
                    small_df = small_df[small_df.index <= settings['right_boundary']]  # обрежем справа

                    if len(small_df) > 0:
                        # если плавный старт и много малых точек
                        if ((small_df[small_df.index == small_df.index[0]].values[0] < 0.8) or
                                (small_df[small_df.index == small_df.index[0]].values[0] >
                                 small_df[small_df.index > settings['left_boundary']].values[0])):
                            small_df = small_df[small_df.index > settings['left_boundary']]

                    amount_of_points_for_each_survey.append(
                        len(small_df))  # запишем количество точек непосредственно для анализа
                    # сделаем разницу между соседними точками
                    values = [0] + list(small_df.values[1::] - small_df.values[0:-1:])
                    # values = [x for x in values if (math.isnan(x) is False)]
                    np_values = np.array(values)

                    np_values_gas = np_values[
                        np_values <= settings['falling_param_val']]  # первая проверка на сильное падение

                    np_values_gas_up = np_values[
                        np_values >= settings['up_down_param_up_val']]  # вторая проверка на падение поменьше
                    np_values_gas_down = np_values[np_values <= settings['up_down_param_down_val']]  # и компенсацию

                    overall_down = np_values.sum()  # третья проверка на итоговое снижение параметра

                    if len(np_values_gas) > 0 or (len(np_values_gas_up) > 0 and len(np_values_gas_down) > 0) or (
                            overall_down < settings['overall_down']):
                        gas_dfs.append(i)  # добавление df с газом
                        gas_periods.append(work_bounds[j])  # добавление интервала с газом
                        overall_down_list.append(overall_down)

                        overall_down_in_percent.append(overall_down / small_df.values[0] * 100)
                        # не максимальное, а первое падение
                        # блок для определения изменения времени работы
                        if len(np_values_gas) > 0:
                            fall_index = np.argwhere(np_values <= settings['falling_param_val'])[0][0]
                        elif len(np_values_gas_up) > 0 and len(np_values_gas_down) > 0:
                            fall_index = np.argwhere(np_values <= settings['up_down_param_down_val'])[0][0]
                        else:
                            fall_index = np.argmin(
                                np_values)  # самое большое падение, возможно первое отрицательное лучше тут
                        fall_index = fall_index - 1  # либо последняя большая точка до падения, либо первая точка после падения
                        fall_dimless_time_index = small_df.index[fall_index]
                        # print(f"fall_index = {fall_index}, fall_dimless_time_index = {fall_dimless_time_index}")
                        qwe = small_df_with_init_index.index[-1]
                        qwe1 = small_df_with_init_index.index[0]
                        this_big_timedelta = datetime.strptime(small_df_with_init_index.index[-1], '%Y/%m/%d %H:%M:%S') \
                                             - datetime.strptime(small_df_with_init_index.index[0], '%Y/%m/%d %H:%M:%S')
                        this_time_to_decrease = (1 - fall_dimless_time_index) * this_big_timedelta
                        this_time_to_decrease = this_time_to_decrease.total_seconds()
                        time_to_decrease_work_period_sec.append(this_time_to_decrease)

                        # привело ли поступление газа к блоку и увеличению накопления, или нет
                        time_to_find_stop = datetime.strptime(work_bounds[j][-1], '%Y/%m/%d %H:%M:%S') + 0.25 * (
                                datetime.strptime(work_bounds[j][-1], '%Y/%m/%d %H:%M:%S') - datetime.strptime(
                            work_bounds[j][0], '%Y/%m/%d %H:%M:%S'))  # время следующего интервала
                        for interval_number, interval in enumerate(stop_bounds):  # пробегаемся по всем интервалам
                            if datetime.strptime(interval[0], '%Y/%m/%d %H:%M:%S') <= time_to_find_stop \
                                    <= datetime.strptime(interval[1], '%Y/%m/%d %H:%M:%S'):
                                if interval_number > 0:
                                    last_stop_timedelta = datetime.strptime(stop_bounds[interval_number - 1][-1],
                                                                            '%Y/%m/%d %H:%M:%S') - \
                                                          datetime.strptime(stop_bounds[interval_number - 1][0],
                                                                            '%Y/%m/%d %H:%M:%S')
                                    this_stop_timedelta = datetime.strptime(interval[-1], '%Y/%m/%d %H:%M:%S') - \
                                                          datetime.strptime(interval[0], '%Y/%m/%d %H:%M:%S')
                                    condition = False
                                    if param == '401':
                                        condition = this_stop_timedelta.total_seconds() \
                                                    / 60 > 2.5 * stats[ei.median_stop_time_by_amperage.param_id]
                                    elif param == '150':
                                        condition = this_stop_timedelta.total_seconds() \
                                                    / 60 > 2.5 * stats[ei.median_stop_time_by_load_engine.param_id]
                                    if (this_stop_timedelta.total_seconds() > last_stop_timedelta.total_seconds() * 2
                                            or condition):
                                        very_bad_dfs.append(i)
                                        very_bad_periods.append(work_bounds[j])

    time_to_decrease_min = np.array(time_to_decrease_work_period_sec) / 60
    if len(time_to_decrease_min) > 3:
        median_time_to_decrease = np.median(time_to_decrease_min)
        mean_time_to_decrease = np.mean(time_to_decrease_min)
        max_time_to_decrease = np.max(time_to_decrease_min)
    else:
        median_time_to_decrease = 0
        mean_time_to_decrease = 0
        max_time_to_decrease = 0
    if len(work_periods) == 0:
        work_periods = [0]

    stats_with_gas = {}
    if param == '401':
        stats_with_gas = {
            ei.supply_gas_by_amperage.param_id: len(gas_periods),
            ei.proportion_of_unstable_periods_by_amperage.param_id: len(gas_periods) / len(work_periods) * 100,
            ei.count_of_blocks_by_amperage.param_id: len(very_bad_periods),
            ei.median_count_for_work_period_by_amperage.param_id: np.median(
                np.array(amount_of_points_for_each_work_period)),
            ei.median_count_for_analysis_by_amperage.param_id: np.median(
                np.array(amount_of_points_for_each_survey)),
            ei.recommended_time_by_amperage.param_id: median_time_to_decrease,
            ei.final_undim_fall_by_amperage.param_id: np.median(np.array(overall_down_list)),
            ei.final_relative_fall_by_amperage.param_id: np.median(
                np.array(overall_down_in_percent)),
        }
    elif param == '150':
        stats_with_gas = {
            ei.supply_gas_by_load_engine.param_id: len(gas_periods),
            ei.proportion_of_unstable_periods_by_load_engine.param_id: len(gas_periods) / len(work_periods) * 100,
            ei.count_of_blocks_by_load_engine.param_id: len(very_bad_periods),
            ei.median_count_for_work_period_by_load_engine.param_id: np.median(
                np.array(amount_of_points_for_each_work_period)),
            ei.median_count_for_analysis_by_load_engine.param_id: np.median(
                np.array(amount_of_points_for_each_survey)),
            ei.recommended_time_by_load_engine.param_id: median_time_to_decrease,
            ei.final_undim_fall_by_load_engine.param_id: np.median(np.array(overall_down_list)),
            ei.final_relative_fall_by_load_engine.param_id: np.median(
                np.array(overall_down_in_percent)),
        }

    stats_with_gas.update(stats)
    # возвращаем блоки для раскраски, а для графиков поступление газа
    return very_bad_periods, gas_dfs, stats_with_gas, df


def find_fail_periods_by_load(this_df, param=gn.motor_load_perc, settings=default_settings):
    """
    Поиск рабочих периодов со значениями параметра, ниже критической - в основом для загрузки ПЭД

    :param this_df: DataFrame с данными
    :param param: параметр для поиска - загрузка по умолчанию
    :param settings: настройки расчеты

    :return: work_periods_with_super_fail_bounds, work_periods_with_fail_df, stats_with_fail, df
            work_periods_with_super_fail_bounds - периоды с критическим значениям параметра и неудачным запуском
            work_periods_with_fail_df - датафреймы с критическим значениям параметра и неудачным запускомstats_with_fail
            df   -  df c  проблемными периодами
    """
    df = this_df.loc[this_df['param_id'] == int(param)].dropna(subset=['val']).sort_values('dt')
    work_status, work_timedelta, \
    stop_timedelta, work_bounds, stop_bounds, stats, _, _ = tools.calculated_regime_time(df, regime_parameter=param,
                                                                                         return_all=True)
    df[gn.work_status_number + f" ({param.replace(' ', '_')})"] = work_status

    crit_value = settings['crit_value_for_load']

    work_periods_with_fail_df = []
    work_periods_with_fail_bounds = []

    work_periods_with_super_fail_df = []
    work_periods_with_super_fail_bounds = []
    for number, k in enumerate(work_bounds):  # последовательно проверим каждый интервал
        this_df = df[(df['dt'] >= k[0]) & (df['dt'] <= k[1])]
        this_df_without_fail = this_df[this_df['val'] >= crit_value]
        if (this_df_without_fail.shape[0] == 0 and  # если больших значений нет
                (number != len(work_bounds) - 1)):  # и интервал не последний;

            # проверка на нулевое значение загрузки: баг сохранения данных в СУ
            next_stop_period_left_border = k[-1]
            next_stop_period_left_border = work_bounds[number + 1][0]
            this_check_df = df[(df['dt'] > next_stop_period_left_border) & (df['dt'] < next_stop_period_left_border)]
            if this_check_df.shape[0] != 1:
                # недогрузка на всем интервале
                work_periods_with_fail_df.append(this_df)
                work_periods_with_fail_bounds.append(k)
                this_timedelta = datetime.strptime(k[-1], '%Y/%m/%d %H:%M:%S') - datetime.strptime(k[0],
                                                                                                   '%Y/%m/%d %H:%M:%S')
                condition = False
                if param == '401':
                    condition = stats[ei.median_work_time_by_amperage.param_id] / 2
                elif param == '150':
                    condition = stats[ei.median_work_time_by_load_engine.param_id] / 2
                # недогрузка привела к остановке
                if this_timedelta.total_seconds() / 60 < condition:
                    work_periods_with_super_fail_df.append(this_df)
                    work_periods_with_super_fail_bounds.append(k)

    stats_with_fail = {}
    if len(work_periods_with_fail_df) != 0:
        stats_with_fail[ei.count_of_starts_with_load_engine_less_than_45.param_id] = len(work_periods_with_fail_df)
    else:
        stats_with_fail[ei.count_of_starts_with_load_engine_less_than_45.param_id] = 0

    if len(work_periods_with_super_fail_df) != 0:
        stats_with_fail[ei.count_of_failed_starts_with_load_engine_less_than_45.param_id] = len(
            work_periods_with_super_fail_df)
    else:
        stats_with_fail[ei.count_of_failed_starts_with_load_engine_less_than_45.param_id] = 0

    return work_periods_with_super_fail_bounds, work_periods_with_fail_df, stats_with_fail, df


def calc_time_carefully(this_df, param=gn.motor_load_perc):
    """
    Функция для расчета рабочего времени ПКВ скважины

    Время работы, время накопления, рабочая доля времени

    :param this_df: DataFrame с данными
    :param param: определяющий параметр (ток, загрузка, частота)
    :param settings: словарь с настройками
    :return: this_df с дополнительными результатми
    """
    df = this_df.loc[this_df['param_id'] == int(param)].dropna(subset=['val']).sort_values('dt')
    work_status, work_timedelta, \
    stop_timedelta, work_bounds, stop_bounds, stats, regime_bounds, regime_timedelta = tools.calculated_regime_time(df,
                                                                                                                    regime_parameter=param,
                                                                                                                    return_all=True)
    df[gn.work_status_number + f" ({param.replace(' ', '_')})"] = work_status
    result_time = []
    result_index = []
    if len(work_bounds) > 0 and len(stop_bounds) > 0:
        # есть какие то интервалы, точно ПКВ #TODO доделать - проблема в None
        if work_bounds[0][0] < stop_bounds[0][0]:
            # первый интервал рабочий
            start = 0
        else:
            # первый интервал не рабочий
            start = 1
        for index in range(start, len(regime_bounds) - 1, 2):
            work_time = regime_timedelta[index]
            stop_time = regime_timedelta[index + 1]
            if work_time != 0 or stop_time != 0:
                active_part_time = work_time / (work_time + stop_time)
                result_time.append(active_part_time)
                result_time.append(active_part_time)
                result_index.append(regime_bounds[index][0])
                result_index.append(regime_bounds[index + 1][1])

        small_df = pd.DataFrame({'Рабочая доля времени': result_time}, index=result_index)

        time_index_work = [x[0] for x in work_bounds]
        work_df = pd.DataFrame({'Время работы, мин': work_timedelta}, index=time_index_work)

        time_index_stop = [x[0] for x in stop_bounds]
        stop_df = pd.DataFrame({'Время накопления, мин': stop_timedelta}, index=time_index_stop)

        return small_df, work_df, stop_df


def find_stucks(df, param=gn.i_a_motor_a):
    """
    Функция для поиска клинов по ПКВ скважине по первому методу

    основная идея при поиске - найти резкие падения параметра,
        и если после большого падения (тока) режим накопления стал больше, то случился клин

    df - подготовленный DataFrame
    param- параметр, по которому будет произодиться поиск клинов, рекомендуется ток

    return -df, stucks, stats
            df - df c колонкой с клинами, если они есть
            stucks - список с временными границами клинов
            stats - статистка по клинами
    """
    start_time = time.time()
    this_file = df.loc[df['param_id'] == int(param)].dropna(subset=['val']).sort_values('dt')

    work_status, work_timedelta, stop_timedelta, work_bounds, stop_bounds, stats, _, _ = tools.calculated_regime_time(
        this_file,
        regime_parameter=param,
        return_all=True,
        last_point_is_working_val=0)
    stucks = []

    this_file = this_file.set_index('dt')
    temp_df = this_file
    temp_df['temp_series'] = [0] + list(
        temp_df['val'].values[1::] - temp_df['val'].values[0:-1])  # работать будем с падением параметра

    time_series = temp_df['temp_series']
    time_series = time_series[(time_series < float('inf')) & (time_series > -float('inf'))]

    not_all_stucks_saved = True

    value = time_series.min()
    while not_all_stucks_saved:
        index = time_series.where(time_series == value).dropna().index[
            0]  # найдет место (время) наибольшего падения для текущего df

        for number, i in enumerate(work_bounds):  # пробежимся по рабочий периодам
            if index >= i[0] and index <= i[1]:  # чтобы найти тот самый рабочий период
                stuck_work_bounds = i
                normal_work_bounds = work_bounds[
                    number - 1]  # предположим, что предыдущий период нормальный, возьмем его границы
                stuck_work_series = time_series[
                    (time_series.index >= stuck_work_bounds[0]) & (
                            time_series.index < stuck_work_bounds[1])]  # создадим series для клина
                normal_work_series = time_series[
                    (time_series.index >= normal_work_bounds[0]) & (
                            time_series.index < normal_work_bounds[1])]  # создадим series для нормальной работы
                stuck_work_timedelta = datetime.strptime(stuck_work_series.index[-1],
                                                         "%Y/%m/%d %H:%M:%S") - datetime.strptime(
                    stuck_work_series.index[0], "%Y/%m/%d %H:%M:%S")  # время клина

                if len(normal_work_series.index) > 0:
                    # время нормальной работы
                    normal_work_timedelta = datetime.strptime(normal_work_series.index[-1],
                                                              "%Y/%m/%d %H:%M:%S") - datetime.strptime(
                        normal_work_series.index[0], "%Y/%m/%d %H:%M:%S")
                    if tools.total_seconds(normal_work_timedelta) != 0:  # период не должен быть пустым
                        if tools.total_seconds(stuck_work_timedelta) / tools.total_seconds(
                                normal_work_timedelta) < 0.25:  # с клином период должен быть меньше

                            this_file.index = pd.to_datetime(this_file.index, format="%Y/%m/%d %H:%M:%S")
                            median_check_df = this_file[
                                (this_file.index >= (datetime.strptime(stuck_work_bounds[0], "%Y/%m/%d %H:%M:%S") - 10 *
                                                     normal_work_timedelta)) &
                                (this_file.index <= (datetime.strptime(stuck_work_bounds[1], "%Y/%m/%d %H:%M:%S") + 10 *
                                                     normal_work_timedelta))]  # возьмем условных 5 соседних интервалов
                            median_check_df = median_check_df[(median_check_df.index < stuck_work_bounds[0]) |
                                                              (median_check_df.index > stuck_work_bounds[
                                                                  1])]  # вырежем интервал с клином
                            max_check_df = this_file[(this_file.index >= stuck_work_bounds[0]) &
                                                     (this_file.index <= stuck_work_bounds[1])]  # сделаем df с клином
                            max_value = max_check_df['val'].max()  # найдем значение клина
                            # if max_value >= median_median_value * 1.5:

                            # добавим границы с клином
                            stucks.append(stuck_work_bounds)
                            # вырежем из исследываемых данных участок с клином
                            time_series = time_series[(time_series.index < i[0]) | (time_series.index > i[1])]

                            # найдем следующее место с наибольшим падением
                            new_value = time_series.min()
        try:
            if new_value != value:
                value = new_value
            else:
                not_all_stucks_saved = False
        except:  # т.к. в первом интервале нету клина, и в последующем не будет
            not_all_stucks_saved = False

    time_spent_sec = time.time() - start_time

    stats = {ei.count_of_stucks_by_amperage.param_id: len(stucks)}

    return df, stucks, stats


def calc_stucks_new(init_df, param=gn.i_a_motor_a):
    """
    Функция для поиска клинов - кратного увеличения параметра (в основном тока)

    :param init_df: стандартный DataFrame с данными
    :param param: определяющий параметр (в основном ток)
    :param settings: настройки расчета
    :return: df, stuck_periods, stats
                df: DataFrame с исходными данными и результатами
                stuck_periods: спискок с периодами времени, для которых был обнаружен клин
                stats: словарь со статистикой
    """
    start_time = time.time()
    df = init_df.loc[init_df['param_id'] == int(param)].dropna(subset=['val']).sort_values('dt')

    work_status, work_timedelta, \
    stop_timedelta, work_bounds, stop_bounds, \
    stats, _, _ = tools.calculated_regime_time(df, regime_parameter=param, return_all=True)

    median_value = tools.calc_median_value(df, work_bounds, left_border=0.3,
                                           right_border=0.7)  # найдем медианное значение параметра

    stuck_periods = []  # создадим маленькие датафреймы для рабочих периодов для безопасной работы с ними
    stuck_df = None

    small_stuck_periods = []  # создадим маленькие датафреймы для рабочих периодов для безопасной работы с ними
    result_small_stuck_df = None

    for k in work_bounds:  # пройдемся по каждому рабочему интервалу
        this_df = df[(df['dt'] >= k[0]) & (df['dt'] <= k[1])]
        # клины
        check_this_df = this_df[this_df['val'] > median_value * 2]
        if check_this_df.shape[0] > 0:  # проверим, есть ли точки, которые значительно больше медианного значения
            stuck_periods.append(k)  # если есть, до добавим текущий период как клин

            # создадим df с клинами
            if stuck_df is None:
                stuck_df = this_df.copy()
            else:
                stuck_df = stuck_df.append(this_df)
        # подклинки
        small_stuck_df = this_df.copy()
        small_stuck_df['dt'] = tools.undim_index(pd.DatetimeIndex(small_stuck_df['dt']))  # обезразмерим по x

        small_stuck_df['val'] = small_stuck_df['val'] / small_stuck_df['val'].max()  # обезразмерим по y

        start_values = small_stuck_df[small_stuck_df['dt'] < 0.3]['val'].values
        middle_values = small_stuck_df[(small_stuck_df['dt'] >= 0.3) &
                                       (small_stuck_df['dt'] <= 0.7)]['val'].values
        if len(start_values) > 0 and len(middle_values) > 0:
            if max(start_values) > max(middle_values) + 0.15:
                small_stuck_periods.append(k)
                if result_small_stuck_df is None:
                    result_small_stuck_df = this_df.copy()
                else:
                    result_small_stuck_df = result_small_stuck_df.append(this_df)

    time_spent = time.time() - start_time

    stats = {ei.median_parameter_value_by_amperage.param_id: median_value,
             ei.count_of_stucks.param_id: len(stuck_periods),
             ei.count_of_substucks.param_id: len(small_stuck_periods)}
    return df, stuck_periods, stats


def find_turb_rotation(check_file, settings=default_settings):
    """
    Определение турбинного вращения

    :param check_file: DataFrame с данными в стандартном формате
    :param settings: словарь с настройками расчетов
    :return: check_file, turb_bounds, stats
                check_file - DataFrame с добавленными случаями турбинного вращения
                turb_bounds - список с временными границами, где было турбинное вращение
                stats - словарь со статистикой
    """
    turb_bounds = []
    stats = {}
    if gn.freq_turb_hz in check_file.columns:
        df = check_file.loc[check_file['param_id'] == int(gn.freq_turb_hz)].dropna(subset=['val']).sort_values('dt')
        df = df.dropna(subset=['val'])
        if df.shape[0] != 0:
            work_status, work_timedelta, stop_timedelta, work_bounds, \
            stop_bounds, stats, regime_bounds, regime_timedelta = tools.calculated_regime_time(df,
                                                                                               regime_parameter=gn.freq_turb_hz,
                                                                                               return_all=True,
                                                                                               last_point_is_working_val=1)
            if len(work_timedelta) != 0:
                for this_number, i in enumerate(work_timedelta):
                    if i > settings['event_turb_rotation_crit_time_min']:
                        j = work_bounds[this_number]
                        turb_bounds.append(j)
                        this_series_with_turb = df[(df['dt'] >= j[0]) & (df['dt'] <= j[1])][gn.freq_turb_hz]
                        this_series_with_turb.name = f"Турбинное вращение по {gn.freq_turb_hz}"
                        if len(turb_bounds) == 1:
                            result_turb_series = this_series_with_turb.copy()
                        else:
                            result_turb_series = result_turb_series.append(this_series_with_turb)

        if len(turb_bounds) > 0:
            check_file = check_file.join(result_turb_series, how='outer')

        stats = {ei.count_of_turb_rotations.param_id: len(turb_bounds)}
    else:
        stats = {ei.count_of_turb_rotations.param_id: -1}
    return check_file, turb_bounds, stats


def analyse_pkv(dataframe, well_id, dt_from, dt_to, connection, parameters=parameters, gn=gn):
    """
    Функция-интерфейс для анализа ПКВ скважин

    выбор частей анализа осуществляется через analysis_parts в settings, если в списке есть
            0 - поступление газа в ЭЦН по току
            1 - поступление газа в ЭЦН по загрузке
            2 - расчет недогрузки
            3 - расчет времени работы и накопления, рабочей доли времени
            4 - поиск клинов методом 1
            5 - поиск клинов методом 2
            6 - поиск турбинного вращения

    :param this_file_name: абсолютный путь до файла с данными
    :param this_file_name_to_save: новый абсолютный путь, куда данные нужно сохранить
    :param parameters: список параметров для отображения на грификах (список из выбранных GlobalNames)
    :param settings: словарь с настройками
    :return: table, df
                table - сводныая таблица с результатами анализа ПКВ в формате DataFrame
                df - DataFrame с добавленными результатами анализа
    """
    start_time = time.time()

    # удаление малых значений по току, которые не несут физического смысла и вызваны способом сохраненения данных в СУ
    dataframe.loc[
        (dataframe['param_id'] == int(gn.i_a_motor_a)) & (dataframe['val'] > 0) & (dataframe['val'] <= 0.5), 'val'] = 0

    stats = {}
    borders = []

    # проверка на режим Чередования Частот (ЧЧ -CF) если он есть, будет преобразовано к стандартному виду ПКВ
    cf_mode, crit_freq = tools.check_cf_mode(dataframe)
    if cf_mode:
        dataframe.loc[(dataframe['param_id'] == int(gn.i_a_motor_a) & (dataframe['val'] == crit_freq)), 0]
        dataframe.loc[(dataframe['param_id'] == int(gn.motor_load_perc) & (dataframe['val'] == crit_freq)), 0]
        stats[ei.work_regime.param_id] = 3
    else:
        stats[ei.work_regime.param_id] = 2

    # расчет поступления газа по току
    gas_borders, gas_dfs, stats_gas, df = find_gas_periods(dataframe, gn.i_a_motor_a)
    stats.update(stats_gas)
    cw.insert_esp_analysis_result(connection, well_id, 0, gas_borders)
    print('Расчет поступления газа по току завершен')

    # расчет поступления газа по загрузке
    gas_borders_load, gas_dfs_load, stats_load, df = find_gas_periods(dataframe, gn.motor_load_perc)
    stats.update(stats_load)
    cw.insert_esp_analysis_result(connection, well_id, 1, gas_borders_load)

    print('Расчет поступления газа по загрузке завершен')

    # расчет недогрузок
    work_periods_with_super_fail_bounds, work_periods_with_fail_df, stats_with_fail, df = find_fail_periods_by_load(
        dataframe, gn.motor_load_perc)
    stats.update(stats_with_fail)
    cw.insert_esp_analysis_result(connection, well_id, 2, work_periods_with_super_fail_bounds)

    print('Расчет недогрузок завершен')

    # расчет времени работы, накопления, активной доли времени
    small_df, work_df, stop_df = calc_time_carefully(dataframe, gn.motor_load_perc)
    print('Расчет времени работы, накопления, активной доли времени завершен')

    # поиск клинов методом 1
    df, stucks, stats_with_stuck = find_stucks(dataframe, param=gn.i_a_motor_a)
    stats.update(stats_with_stuck)
    cw.insert_esp_analysis_result(connection, well_id, 3, stucks)
    print("Поиск клинов методом 1 завершен")

    df, stuck_periods3, stats3 = calc_stucks_new(dataframe, param=gn.i_a_motor_a)
    stats.update(stats3)
    cw.insert_esp_analysis_result(connection, well_id, 4, stuck_periods3)
    print("Поиск клинов методом 2 завершен")

    df, turb_bounds, turb_stats = find_turb_rotation(dataframe)
    stats.update(turb_stats)
    cw.insert_esp_analysis_result(connection, well_id, 5, turb_bounds)
    print("Поиск турбинного завершен")

    end_time = time.time()
    stats[ei.spent_total_time.param_id] = end_time - start_time

    cw.insert_esp_analysis_result_common(connection, well_id, dt_from, dt_to, stats)

    # построение графиков
    banches = pltl_wf.create_banches_for_report(df, parameters, fuzzy_names=True)

    stats['Ссылка на файл'] = f"file:///{this_file_name_to_save}"

    table = pd.DataFrame(stats, index=[this_file_name.split('\\')[-1]])

    table['Время начала записи'] = [df.index[0]]
    table['Время конца записи'] = [df.index[-1]]
    table['Количество записей'] = [df.shape[0]]

    table_for_plot = table.T
    table_for_plot.index.name = 'Название исходного файла'

    pltl_wf.create_report_html(df, banches, this_file_name_to_save, borders=borders, auto_open=False,
                               df_for_table=table_for_plot)
    if settings['data_save_analysis_in_csv'] == 1:
        dir_name_with_csv = 'analysis_results_pkv_csv'
        work_dir = settings['work_dir']
        Path(f"{work_dir}\\{dir_name_with_csv}").mkdir(parents=True, exist_ok=True)
        file_name_csv = this_file_name_to_save.replace('.html', '.csv')
        file_name_csv = file_name_csv.replace('analysis_results_pkv', dir_name_with_csv)
        df.to_csv(file_name_csv)

    return table, df
