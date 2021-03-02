import sys

import pandas as pd
import numpy as np
import datetime as dt
import enum


class EspIndex(enum.Enum):
    supply_gas_by_amperage = (6, 'Поступление газа по току фазы А')
    proportion_of_unstable_periods_by_amperage = (7, 'Доля нестабильных режимов по току фазы А')
    count_of_blocks_by_amperage = (8, 'Количество блоков из-за газа по току фазы А')
    median_count_for_work_period_by_amperage = (9, 'Медианное количество точек для периода работы по току фазы А')
    median_count_for_analysis_by_amperage = (10, 'Медианное количество точек для анализа по току фазы А')
    recommended_time_by_amperage = (11, 'Рекомендуемое (медианное) время уменьшения работы по току фазы А')
    final_undim_fall_by_amperage = (12, 'Медианное итоговое падение параметра по току фазы А, (обезразм), д.ед.')
    final_relative_fall_by_amperage = (13, 'Медианное итоговое падение параметра по току фазы А, (относит), %')
    median_work_time_by_amperage = (14, 'Медианное время работы по току фазы А, мин')
    median_stop_time_by_amperage = (15, 'Медианное время накопления по току фазы А, мин')
    work_proportion_by_amperage = (16, 'Рабочая доля времени за весь период по току фазы А')
    big_stop_count_by_amperage = (17, 'Количество больших остановок по току фазы А')
    supply_gas_by_load_engine = (18, 'Поступление газа по загрузке двигателя, раз')
    proportion_of_unstable_periods_by_load_engine = (19, 'Доля нестабильных режимов по загрузке двигателя, %')
    count_of_blocks_by_load_engine = (20, 'Количество блоков из-за газа по загрузке двигателя, раз')
    median_count_for_work_period_by_load_engine = \
        (21, 'Медианное количество точек для периода работы по загрузке двигателя, штук')
    median_count_for_analysis_by_load_engine = (22, 'Медианное количество точек для анализа по загрузке двигателя')
    recommended_time_by_load_engine = \
        (23, 'Рекомендуемое (медианное) время уменьшения работы по загрузке двигателя, мин')
    final_undim_fall_by_load_engine = \
        (24, 'Медианное итоговое падение параметра по загрузке двигателя, (обезразм) д.ед.')
    final_relative_fall_by_load_engine = \
        (25, 'Медианное итоговое падение параметра по загрузке двигателя, (относит) %.')
    median_work_time_by_load_engine = (26, 'Медианное время работы по загрузке двигателя, мин')
    median_stop_time_by_load_engine = (27, 'Медианное время накопления по загрузке двигателя, мин')
    work_proportion_by_load_engine = (28, 'Рабочая доля времени за весь период по загрузке двигателя, %')
    big_stop_count_by_load_engine = (29, 'Количество больших остановок по загрузке двигателя, %')
    count_of_starts_with_load_engine_less_than_45 = (30, 'Количество запусков с Загрузкой меньше 45, %')
    count_of_failed_starts_with_load_engine_less_than_45 = \
        (31, 'Количество неудачных запусков с Загрузкой меньше 45, %')
    count_of_stucks_by_amperage = (32, 'Количество клинов (первым методом)), штук')
    median_parameter_value_by_amperage = (33, 'Режимное значение параметра по току фазы А')
    count_of_stucks = (34, 'Количество клинов (вторым методом), штук')
    count_of_substucks = (35, 'Количество подклинок, штук')
    count_of_turb_rotations = (36, 'Число случаев турбинного вращения')
    spent_total_time = (37, 'Затрачено времени на анализ, сек')
    work_regime = (38, 'Режим работы')

    def __init__(self, param_id, val):
        self.param_id = param_id
        self.val = val


class GlobalNames():
    """
    Класс для хранения идентификаторов параметров, используемых при анализе, а также приведения их к одному типу
    """
    def __init__(self):
        self.q_liq_m3day = '12'  # Дебит жидкости, м3/сут
        self.q_gas_m3day = '204'  # Дебит газа, м3/сут
        self.q_oil_mass_tday = '30'  # Дебит нефти массовый, т/сут
        self.watercut_perc = 'Обводненность, %'
        self.gor_m3m3 = 'ГФ, м3/м3'
        self.p_buf_atm = 'Буферное давление, атм'
        self.p_lin_atm = '142'  # Линейное давление, атм
        self.p_intake_atm = 'Давление на приеме, атм'
        self.t_intake_c = 'Температура на приеме, С'
        self.cos_phi_d = '3001'  # Коэффициент мощности, д.ед.
        self.u_motor_v = 'Напряжение на выходе ТМПН, В'
        self.i_a_motor_a = '401'  # Ток фазы А, А
        self.motor_load_perc = '150'  # Загрузка двигателя
        self.freq_hz = '220'  # Частота вращения, Гц
        self.active_power_kwt = '403'  # Активная мощность, кВт
        self.c_calibr_head_d = 'К. калибровки по напору - множитель, ед'
        self.c_calibr_power_d = 'К. калибровки по мощности - множитель, ед'
        self.t_motor_c = '187'  # Температура двигателя
        self.efficiency_esp_d = 'КПД ЭЦН, д.ед.'
        self.esp_head_m = 'Напор ЭЦН, м'
        self.dp_esp_atm = 'Перепад давления ЭЦН, атм'
        self.vibration_xy_msec2 = 'Вибрация X/Y, м/с2'
        self.vibration_z_msec2 = 'Вибрация X/Y, м/с2'
        self.work_status_number = 'Номер режима работы'
        self.freq_turb_hz = 'Частота турбинного вращения, Гц'
        self.full_power_kwt = 'Полная мощность, кВт'
        self.u_ab_v = 'Напряжение AB, В'
        self.q_oil_m3day = 'Дебит нефти, м3/сут'
        self.current_dt = 'dt'  # Время

    def return_dict_column_to_rename(self):
        columns_name_to_rename = {
            self.active_power_kwt: ["Активная мощность", 'Ракт, кВт',
                                    "Активная мощность (ТМ)", 'Pa,кВт', 'акт.P,кВт', 'Pакт(кВт)',
                                    'Активная мощность, кВт',
                                    'Активная мощность, кВт'],
            self.full_power_kwt: ["Pполн,кВт", 'P, кВА', 'Pполн(кВA)', 'Мощность активная, КВт'],
            self.p_lin_atm: ["Линейное давление", "Давление линейное (ТМ)"],

            self.p_intake_atm: ["Давление на приеме насоса (пласт. жидкость)",
                                "Давление на входе ЭЦН (ТМ)",
                                'P на приеме,ат',
                                'P, ат.', 'P,atm', 'Pвх(МПа)', 'Pнас, кгс/см2',
                                'Давление на приеме насоса', 'Рокр, ат',
                                'Давление жидкости на приеме насоса , Атм',
                                'Давление на приеме насоса, атм'],
            self.t_intake_c: ["Температура насоса ЭЦН (ТМ)", "Температура на приеме насоса (пласт. жидкость)",
                              "Тжид,Гр", 'Тнас, С',
                              'Tжид, °C', 'Твх(°С)',
                              'Tокр, °C', 'Температура ПЭД, °C'],

            self.t_motor_c: ["Температура двигателя ЭЦН (ТМ)", "ТПЭД,Гр", 'Tдвиг, °C',
                             'Тобм(°С)', 'Тэд, С',
                             'Температура двигателя.1', 'ТобмПЭД, °C', 'Температура двигателя , °С'],

            self.motor_load_perc: ["Загрузка двигателя", 'Загр,%',
                                   "Загрузка ПЭД (ТМ)", "Загр,%", 'Загр., %', 'Загр., %',
                                   'Загрузка,%', 'Загр(%)', 'Загрузка, %', 'Загрузка', 'Загр. дв, %',
                                   'Загр. ПЭД, %', 'Загрузка ПЭД, %',
                                   'Загрузка , %', 'Загр., %'],
            self.u_ab_v: ["Входное напряжение АВ", "Напряжение AB (ТМ)", "UAB,В", 'Uвх.AB,В', 'Uab,В',
                          'UвхAB(B)', 'Uab, В', 'Напряжение BA', 'Uвх.AB, В',
                          'Напряжение входа AB, В', 'Напряжение AB, B'],
            self.i_a_motor_a: ["Ток фазы А", "Ток фазы A (ТМ)", "Ia,А", 'Ia, A', 'Iа(A)', 'Ia, А',
                               'Ток фазы A', "Ia, A", 'Ia ПЭД, А',
                               'Ток выходной фаза U, А',
                               'Ток фазы A, A'],
            self.freq_hz: ["Выходная частота ПЧ", "Частота вращения (ТМ)", "F,Гц", 'F, Гц', 'F(Гц)',
                           'Fвр, Гц', 'Частота вращения двигателя', "F, Гц", 'Fтек, Гц',
                           'Частота вращения ПЭД (Гц), Гц', 'Частота вращения , Гц'],

            self.freq_turb_hz: ['Fтур, Гц', 'F Турб.вращ.,Гц'],
            self.cos_phi_d: ["Коэффициент мощности", "Коэффициент мощности (ТМ)", "Cos", 'cos',
                             'Коэффициент мощности', 'Коэффициент мощности, д.ед.',
                             'Cos ф'],

            self.q_liq_m3day: ["Объемный дебит жидкости", "Дебит жидкости (ТМ)"],
            self.q_gas_m3day: ["Объемный дебит газа", "Дебит газа (ТМ)"],
            self.watercut_perc: ["Процент обводненности", "Обводненность (ТМ)"],
            self.q_oil_m3day: ["Объемный дебит нефти"],
            self.q_oil_mass_tday: ["Дебит нефти (ТМ)"],
            self.vibration_xy_msec2: ['Вибр X/Y, м/с2'],
            self.vibration_z_msec2: ['Вибр Z, м/с2'],
            self.work_status_number: ['work_status'],
            self.current_dt: ['Время']}

        return columns_name_to_rename

    def renamed_parameters(self):
        renamed_parameters = [self.q_liq_m3day,
                              self.q_gas_m3day,
                              self.q_oil_mass_tday,
                              self.p_lin_atm,
                              self.cos_phi_d,
                              self.i_a_motor_a,
                              self.freq_hz,
                              self.active_power_kwt,
                              self.t_motor_c,
                              self.current_dt,
                              self.motor_load_perc]
        return renamed_parameters


def rename_columns_by_dict(df, columns_name_dict):
    for i in df.columns:
        for items in columns_name_dict.items():
            if i in items[1] or i in [x.replace(' ', '') for x in items[1]]:
                df = df.rename(columns={i: items[0]})
    return df


def get_well_id(file_name):
    """
    Возвращает айди скважины по имени файла csv
    """
    df = pd.read_csv('C:\\Users\Damirchik\\Documents\\test.csv', encoding='cp1251')

    return df[df['fileName'] == file_name]['id'].values[0]


def calculated_regime_time(dataframe, regime_parameter, return_all=False, last_point_is_working_val=1):
    """
    Базовая функция для расчета параметров работы ПКВ
    Определение периодов работы и накопления с первичной статистикой
    :param dataframe: DataFrame с данными
    :param regime_parameter: параметр, по которому будет определяться рабочий режим (обычно ток или загрузка)
    :param return_all: флаг для возврата всех результатов
    :param last_point_is_working_val: 1 - последняя точка рабочего интервала больше нуля, 0 - последняя точка будет равна нулю
    :return
        if not return_all:
            return work_time_median, stop_timedelta_median
        else:
            return work_status, work_timedelta, stop_timedelta,
                work_bounds, stop_bounds, stats, regime_bounds, regime_timedelta

                work_status - список из 1 и 0 - 1 работает, 0 - нет
                work_timedelta - список из длительности каждого включения в минутах
                stop_timedelta - список из длительности каждого накопления в минутах
                work_bounds - список из временных границ для каждого рабочего режима
                stop_bounds - список из временных границ для каждого режима накопления
                stats - словарь с первичной статистикой ПКВ
                regime_bounds - список из временных границ для работы и накопления
                regime_timedelta - список из длительности работы и накопления в минутах


    """
    last_time = dataframe['dt'].iloc[0]

    work_status = []
    if dataframe['val'].iloc[0] > 0:
        work_status.append(1)
    else:
        work_status.append(0)

    work_timedelta = []
    stop_timedelta = []
    regime_bounds = []
    regime_timedelta = []
    work_bounds = []
    stop_bounds = []

    for i in range(1, dataframe.shape[0]):  # пробежимся по временному ряду
        this_time = dataframe['dt'].iloc[i]
        this_value = dataframe['val'].iloc[i]
        if this_value > 0:
            work_status.append(1)  # точка рабочая
        else:
            work_status.append(0)  # точка нерабочая
        # print(this_value)
        time_delta = dt.datetime.strptime(this_time, '%Y/%m/%d %H:%M:%S') - dt.datetime.strptime(last_time, '%Y/%m/%d %H:%M:%S')
        # time_delta = this_file.index[i-1] - last_time
        if work_status[-1] != work_status[-2]:

            # print('Переключение')
            if work_status[-1] == 1:
                # print('Включение')
                stop_timedelta.append(time_delta.total_seconds() / 60)
                stop_bounds.append([last_time, this_time])

                regime_timedelta.append(time_delta.total_seconds() / 60)
                regime_bounds.append([last_time, this_time])
            else:
                # print('Выключение')
                this_time = dataframe['dt'].iloc[i - last_point_is_working_val]  # возьмем предыдущую точку (ненулевую)
                buf_time = dataframe['dt'].iloc[i]  # нулевая точка для последнего периода
                time_delta = dt.datetime.strptime(this_time, '%Y/%m/%d %H:%M:%S') - dt.datetime.strptime(last_time, '%Y/%m/%d %H:%M:%S')

                work_timedelta.append(time_delta.total_seconds() / 60)
                work_bounds.append([last_time, this_time])

                regime_timedelta.append(time_delta.total_seconds() / 60)
                regime_bounds.append([last_time, this_time])

            last_time = this_time
    if len(set(work_status)) != 1:
        if work_status[-1] == 1:
            # print('Последний интервал работала')
            work_timedelta.append(time_delta.total_seconds() / 60)
            work_bounds.append([last_time, this_time])
        else:
            # print('Последний интервал не работала')
            stop_timedelta.append(time_delta.total_seconds() / 60)
            stop_bounds.append([buf_time, this_time])
    else:  # если был только 1 режим
        if work_status[-1] == 1:
            # print('Весь интервал работала')
            work_timedelta.append(time_delta.total_seconds() / 60)
            work_bounds.append([last_time, this_time])
        else:
            # print('Весь интервал не работала')
            stop_timedelta.append(time_delta.total_seconds() / 60)
            stop_bounds.append([last_time, this_time])

    work_time_median = np.median(work_timedelta)
    stop_timedelta_median = np.median(stop_timedelta)
    work_fraction_perc = np.sum(work_timedelta) / (np.sum(work_timedelta) + np.sum(stop_timedelta)) * 100
    stop_timedelta_np_array = np.array(stop_timedelta)
    amount_of_big_stops = len(stop_timedelta_np_array[stop_timedelta_np_array > stop_timedelta_median * 2])

    stats = {}
    if regime_parameter == '401':
        stats = {EspIndex.median_work_time_by_amperage.param_id: work_time_median,
                 EspIndex.median_stop_time_by_amperage.param_id: stop_timedelta_median,
                 EspIndex.work_proportion_by_amperage.param_id: work_fraction_perc,
                 EspIndex.big_stop_count_by_amperage.param_id: amount_of_big_stops}
    elif regime_parameter == '150':
        stats = {EspIndex.median_work_time_by_load_engine.param_id: work_time_median,
                 EspIndex.median_stop_time_by_load_engine.param_id: stop_timedelta_median,
                 EspIndex.work_proportion_by_load_engine.param_id: work_fraction_perc,
                 EspIndex.big_stop_count_by_load_engine.param_id: amount_of_big_stops}

    if not return_all:
        return work_time_median, stop_timedelta_median
    else:
        return work_status, work_timedelta, stop_timedelta, work_bounds, stop_bounds, stats, regime_bounds, regime_timedelta


def check_cf_mode(dataframe):
    """
    Определение режима работы ЧЧ - чередования частот

    :param file: DataFrame с данными в стандартном форматер
    :return: True - это ПКВ с ЧЧ, False - это не ПКВ с ЧЧ
    """
    cf_mode = False
    crit_freq = -1

    popular_freq = dataframe[dataframe['val'] > 0]['val'].value_counts().index.values[:2]
    if len(popular_freq) > 0:
        crit_freq = popular_freq.min()
        max_freq = popular_freq.max()

        values = dataframe['val'].values
        fr_values_check = np.array([0] + list(values[1::] - values[0:-1:]))
        amount_of_stops = len(fr_values_check[fr_values_check <= -crit_freq])
        values_without_cf = np.where(
            (dataframe['val'] >= crit_freq - 1) & (dataframe['val'] <= crit_freq + 1),
            0,
            dataframe['val'])
        fr_values_check_without_cf = np.array([0] + list(values_without_cf[1::] - values_without_cf[0:-1:]))
        amount_of_stops_without_cf = len(fr_values_check_without_cf[fr_values_check_without_cf <= -crit_freq])

        if (amount_of_stops_without_cf > amount_of_stops * 5) and (max_freq - crit_freq) >= 3:
            print('Это ЧЧ')
            print(f"Частота холостого хода: {crit_freq} Гц")
            cf_mode = True

    return cf_mode, crit_freq


def get_work_mode(df, well_id, motor_load_percent=150, freq_hz=220) -> bool:
    """
    Функция для определения режима работы скважины

    return 0 — пкв
    return 1 - пдф
    """
    try:
        df_to_check_pkv = df.loc[df['param_id'] == int(motor_load_percent)].dropna(subset=['val']).sort_values('dt')
        df_to_check_cf = df.loc[df['param_id'] == int(freq_hz)].dropna(subset=['val']).sort_values('dt')

        work_status, work_timedelta, stop_timedelta, work_bounds, stop_bounds, stats, _, _ = calculated_regime_time(
            df_to_check_pkv,
            regime_parameter=motor_load_percent,
            return_all=True)
        cf_mode, _ = check_cf_mode(df_to_check_cf)

        if ((
                (stats[EspIndex.work_proportion_by_load_engine.param_id] > 80) or
                (len(stop_timedelta) < 15) or
                (stats[EspIndex.median_stop_time_by_load_engine.param_id] == np.median([])) or
                (stats[EspIndex.median_stop_time_by_load_engine.param_id] >= 300) or
                (df_to_check_pkv['val'].min() >= 10)
        )
                and (not cf_mode)
        ):  # медианное время работы
            print(f"ПДФ режим для {well_id}")
            return 1
        else:
            print(f"ПКВ режим для {well_id}")
            return 0
    except Exception as e:
        print(e)


def undim_index(time_index):
    """
    Функция для обезразмеривания временного индекса и приведения его к формату [0..1]

    time_index - DataFrame.index c форматов DateTime

    return обезмеренный time_index
    """
    time_index = time_index - time_index[0]
    time_index = time_index.total_seconds()
    time_index = time_index / time_index[-1]
    return time_index


def get_true_median_value_in_series(df, column, except_zero=True):
    """
    Функция для получения медианного значения параметра в ПКВ режиме с учетом неравномерности сохранения данных

    :param df: DataFrame с данными
    :param column: название колонки с параметрами
    :param except_zero: флаг исключения нулевых значений
    :return: правильное медианное значение параметра
    """
    # сделаем чистый series с равномерным заполнением
    this_df = df.copy()
    this_series = this_df['val']
    this_series = this_series.resample('1s').mean()
    this_series = this_series.interpolate('linear')
    # уберем ненужные нули при необходимости
    if except_zero:
        this_series = this_series[this_series > 0]

    # найдем медианное значение
    median_value = this_series.median()
    if df['val'].median() > median_value:  # возьмем большее из двух (без интерполяции и с оной)
        median_value = df[column].median()

    top_values_len = df[df['val'] > df['val'].max() * 0.9].shape[0]
    df['dt'] = df.index
    df['dt'] = df['dt'].dt.strftime('%Y/%m/%d %H:%M:%S').astype(str)
    _, work_timedelta, _, _, _, _, _, _ = calculated_regime_time(df, regime_parameter=column, return_all=True,
                                                                 last_point_is_working_val=0)
    if len(work_timedelta) * 2 < top_values_len:
        median_value = df[df['val'] > df['val'].max() * 0.9]['val'].median()
    return median_value


def total_seconds(timedelta):
    try:
        seconds = timedelta.total_seconds()
    except AttributeError:  # no method total_seconds
        one_second = np.timedelta64(1000000000, 'ns')
        # use nanoseconds to get highest possible precision in output
        seconds = timedelta / one_second
    return seconds


def calc_median_value(df, work_periods, left_border=0.3, right_border=0.7):
    df = df.set_index('dt')
    param_series = df['val']

    values = []

    for left_work_boundary, right_work_boundary in work_periods:
        small_series = param_series[(param_series.index >= left_work_boundary) &
                                    (param_series.index <= right_work_boundary)]
        small_series.index = undim_index(pd.DatetimeIndex(small_series.index))
        small_series_med = small_series[(small_series.index >= left_border) &
                                        (small_series.index <= right_border)]
        if len(small_series_med) == 0:
            small_series_med = small_series[(small_series.index >= left_border - 0.1) &
                                            (small_series.index <= right_border + 0.1)]
            if len(small_series_med) == 0:
                small_series_med = small_series
        values = values + small_series_med.tolist()

    return np.median(np.array(values))

