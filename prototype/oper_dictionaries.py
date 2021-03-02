'''
Функции для создания словарей
'''
from itertools import groupby
import csv

def create_dict_x_last_with_statistic_accuracy(dict_with_list_of_predict_opers):
    '''
    Расчет статистической вероятности в словаре, в котором значения по ключу расположены в виде списка, вида {key: [n, n2, n3, n4, n5, ...]}
        1. Находим количество вхождения в список каждого элемента
        2. Находим элемент, у которого количество вхождения макисмально
        3. Частота появления этого элемента: элемент / длину списка
        4. Создаем новый словарь {ключ_операци: статистическая вероятность}
    :param dict_with_list_of_predict_opers: словарь вида {key_oper: [n, n2, n3, n4, n5, ...]}
    :return: словарь вида {ключ_операции: статистическая вероятность}
    '''
    dict_x_last_with_statistic_accuracy = {}
    statistic_value = 0
    for key, value in dict_with_list_of_predict_opers.items():
        if isinstance(value, list):
            count_of_elements = [len(list(group)) for key, group in groupby(value)]  # выводит количество повторений каждой операции в списке
            print(count_of_elements)
            max_count_of_elements = max(count_of_elements)  # поиск максимального числа среди всех повторяющихся
            print(max_count_of_elements)
            statistic_value = max_count_of_elements / len(value)  # статистическая вероятность
            print(statistic_value)
        dict_x_last_with_statistic_accuracy[key] = statistic_value # создание нового словаря вида {ключ_операции: статистическая вероятность}
    return dict_x_last_with_statistic_accuracy


def dict2code(code2_dict, dict_with_data):
    '''
    Функция для создания словаря с реальным идентификатором операции и значением точности (либо статистической, либо предсказания)

    :param code2_dict: cловарь, в котором ключи - реальные идентификаторы операций {real_id : code_id}
    :param dict_with_data: словарь, в котором значения по ключу - значения точности  {code_id : accuracy)
    :return: словарь dict2code, в котором ключ - реальный идентификатор, значение по ключу - соответсвующая точность {real_id : accuracy}
    '''

    dict2code = code2_dict.copy()
    for key, value in dict_with_data.items():
        dict2code[key] = dict_with_data.get(key)
    return dict2code


def create_dict_with_statistic_and_predict_accuracy(statistic_accuracy, predict_accuracy):
    '''
    Создание словаря вида {ключ операции : [точность предсказания сетью, статистическая точность]}
    :param statistic_accuracy: результат работы функции create_dict_x_last_with_statistic_accuracy
    :param predict_accuracy: рузльатт работы функции predict - dict_x_last_with_predict_accuracy
    :return:
    '''
    dict = {}
    for key in predict_accuracy:
            dict[key] = []
            dict[key].append(predict_accuracy.get(key)) # добавление в список вероятности предсказания нейросетью
            dict[key].append(statistic_accuracy.get(key)) # добавление в список статистической вероятности предсказания
    return dict

def create_txt(data):
    '''
    Запись словаря в txt файл
    '''
    with open('txt.txt', 'w') as out:
        for key, val in data.items():
            out.write('{}:{}\n'.format(key, val))

def create_csv(data):
    '''
    Запись словаря в csv файл
    '''
    with open("csv.csv", "w") as f:
        writer = csv.writer(f)
        for i in data:
          writer.writerow([i, data[i]])
    f.close()