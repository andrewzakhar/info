#
# Формирование наборов данных.
#

# %%

import numpy as np
import math

def split_to_train_and_test (ds, train_part: float, shuffle: bool):
    """"
    Рандомное деление на проверочные и тестовые данные
    """
    if shuffle:
        np.random.shuffle(ds)

    data_m = ds.shape[0]

    train_m = int(data_m * train_part)
    test_m = len(ds) - train_m

    print(train_m, test_m)

    seq_len = ds.shape[1]
    train_x = ds[:train_m, :seq_len - 1].reshape(train_m, seq_len - 1)
    train_y = ds[:train_m, seq_len - 1:].reshape(train_m, 1)
    test_x = ds[train_m:, :seq_len - 1].reshape(test_m, seq_len - 1)
    test_y = ds[train_m:, seq_len - 1:].reshape(test_m, 1)

    return (train_x, train_y, test_x, test_y)

def split_to_train_and_test_with_date(ds, date):
    """"
    Деление на тестовые и проверочные наборы данных по дате:
    меньше выбранной даты - train, больше - test
    """
    data_train = []
    data_test = []
    ds = ds.tolist()
    for row in range(len(ds)):
        for col in range(len(ds[row])):
            if ds[row][1] < date:  # проверка даты
                del ds[row][1] # удаление столбца с датой
                data_train.append(ds[row])
            else:
                del ds[row][1]
                data_test.append(ds[row])
            break

    # Деление на проверочные и тестовые наборы данных
    data_train = np.array(data_train)  # list -> np.array для применения методаа np.reshape
    data_test = np.array(data_test)

    train_row = data_train.shape[0]
    test_row = data_test.shape[0]
    seq_len = data_test.shape[1]

    train_x = data_train[:train_row, :seq_len - 1].reshape(train_row, seq_len - 1)
    train_y = data_train[:train_row, seq_len - 1:].reshape(train_row, 1)
    test_x = data_test[:test_row, :seq_len - 1].reshape(test_row, seq_len - 1)
    test_y = data_test[:test_row, seq_len - 1:].reshape(test_row, 1)

    return (train_x, train_y, test_x, test_y)


def repairs_types_filter (repairs, min_m):
    # Считаем количество ремонтов каждого из типов.
    data_unique_repairs, data_unique_repairs_count = np.unique(repairs[:, 0], return_counts=True)
    # print(data_unique_repairs, data_unique_repairs_count)
    # Выбираем коды ремонтов, по которым более заданного числа ремонтов есть.
    repairs_types = np.take(data_unique_repairs, np.where(data_unique_repairs_count > min_m))[0]

    return repairs_types


def get_windowed_data(repairs, win_size, p_count):
    """
    Возвращает numpy-массив разрезанных "окнами" данных.
    """
    # win_size - какое количество операций входит в одно окно
    r_with_date = 2  # тип ремонта + дата
    step = p_count + 1
    new_win_size = r_with_date + step * win_size + 1  # величина нового окна, 1 - для записи следующей операции (для разделения на train и test)

    data = np.empty((0, new_win_size), dtype='int32')

    for r_ind in range(0, len(repairs)):
        repair = repairs[r_ind]

        if len(repair) < new_win_size:
            continue

        data_item = np.zeros((new_win_size), dtype='int32')

        for i in range(0, new_win_size):
            data_item[i] = repair[i]
        data = np.append(data, np.array([data_item]), axis=0)

        oper_count = (len(repair) - 2) / step
        win_count = math.floor(oper_count - win_size)  # количество заполняемых окон
        for i in range(1, win_count):
            for j in range(2, new_win_size):
                data_item[j] = repair[i * step + j]
            data = np.append(data, np.array([data_item]), axis=0)

    return data

def get_net_inputs(data, param_count):
    ''''
    Разделение на отдельные входы операций, параметров и значений параметров
    '''
    oper_input = []
    oper_param_input = []
    param_value_input = []
    for row in range(len(data)):
        for opers in range(1, len(data[row]), param_count + 1):
            # Заполнение массива операций
            oper_input.append(data[row][opers])

            # Заполнение массива параметров
            for j in range(1, param_count, 2):
                oper_param_input.append(data[row][opers + j])

            # Заполнение массива значений параметров
            for j in range(2, param_count + 1, 2):
                param_value_input.append(data[row][opers + j])

    return oper_input, oper_param_input, param_value_input