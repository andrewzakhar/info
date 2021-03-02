# %%
# Метод оценки вероятности операций из N
# Расчет статистической точности
#

# Метод оценки вероятности операций из N

import numpy as np
from itertools import groupby
import oper_loader_and_encoder


def calc_accurracy(model, test_x, test_y, predict_len):
    """
    Расчет для каждой операции x_last вероятности того, что хотя бы один из следующих predict_len предсказанных элементов
    после последовательности длиной seq_len (для которой обучена модель), заканчивающейся на эту операцию x_last,
    совпадает со следующим после x_last элементом в тестовой выборке - test_y.
    """

    print('Predicting...')
    x_last_count = {}  # количество x_last
    y_1 = {}  # y_1[x_last]
    accuracy = {}

    for x_index in range(0, len(test_x), 1):
        x = test_x[x_index:(x_index + 1)]
        x_last = test_x[x_index][-1]
        if x_last not in x_last_count:
            x_last_count[x_last] = 0
        x_last_count[x_last] += 1
        if x_last not in y_1:
            y_1[x_last] = [0] * predict_len
        for predict_index in range(predict_len):
            y_pred = model.predict_classes(test_x[x_index:(x_index + 1)])
            if y_pred == test_y[x_index]:
                for j in range(predict_index, predict_len):
                    y_1[x_last][j] += 1
                break
            x = np.append(x, y_pred)
            x = x[1:]  # сдвиг окна на шаг вправо
            test_x[x_index:(x_index + 1)] = x  # для предикта трех разных значений [1 2 3]

    for x_last in y_1:
        accuracy[x_last] = [0] * predict_len
        for i in range(predict_len):
            accuracy[x_last][i] = y_1[x_last][i] / x_last_count[x_last]
    return accuracy, x_last_count


# %%

# Вывод операции и следующих после нее операций
def calc_next_opers(repairs, code2_dict):
    next_opers = {k: [] for k in code2_dict}
    for i in range(len(repairs)):
        for j in range(len(repairs[i]) - 1):
            if repairs[i][j] in code2_dict:
                next_opers[repairs[i][j]].append(repairs[i][j + 1])
    return next_opers


# %%

# Расчет статистической точности
def calc_stat_accuracy(next_opers):
    statisctic_acc = {}
    for key, value in next_opers.items():
        value = oper_loader_and_encoder.quicksort(value)  # сортировка по возрастанию
        count_of_elements = [len(list(group)) for key, group in
                             groupby(value)]  # количество повторяющихся элементов в списке
        statistic_value = max(count_of_elements) / len(
            value)  # частота появления максимально часто встречающегося значения в списке
        statisctic_acc[key] = statistic_value
    return statisctic_acc
