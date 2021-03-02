
import numpy as np
import matplotlib.pyplot as plt
import json

import oper_loader_and_encoder
import oper_preprocessor

out_opers_code_path = 'data/out_opers_code.csv'
out_params_code_path = 'data/out_opers_code.csv'
out_params_values_code_path = 'data/out_params_values_code.csv'
out_repairs_code_path = 'data/out_repairs_code.csv'
out_code2_path = 'data/out_code2.csv'

seq_len = 5
repairs_windowed_dataset_path = 'data/out_repairs_windowed' + str(seq_len) + '.npy'

def prepare_data():
    print('Loading...')
    repairs = oper_loader_and_encoder.load_svod(
        'data/20191202_dump__svod_opers_v.csv',
        'data/20191202_dump__svod_opers_parms_v.csv',
        'data/20191202_dump__repairs_v.csv',
        True,
        [],
        False,
        out_opers_code_path,
        out_params_code_path,
        out_params_values_code_path,
        out_repairs_code_path)

    # Сохранение загруженного и закодированного с помощью метода load_svod массива данных (repairs) в файл (data/out_repairs.csv).
    oper_loader_and_encoder.dump_loaded_svod_data(repairs, 'data/out_repairs.csv')

    #repairs = oper_loader.load_svod_data_dump('data/out_repairs.csv')
    repairs = np.array([np.array(r) for r in repairs]) #[:1000]

    # Возвращает numpy-массив разрезанных "окнами" данных.
    repairs_windowed = oper_preprocessor.get_windowed_data(repairs, seq_len)
    np.save(repairs_windowed_dataset_path, repairs_windowed)

    print('End of loading.')

# prepare_data() #раскомменчивать только при первом запуске
ds = np.load(repairs_windowed_dataset_path)
#print(ds)

# Типы ремонтов, по которым есть более указанного числа последовательностей min_m.
repairs_type_filter = oper_preprocessor.repairs_types_filter(ds,30) # возврат типов

#repairs_type_filter = oper_preprocessor.repairs_type_17(ds) # возврат типа 17
# print(repairs_type_filter)

# Выбор из всех типов ремонтов только наиболее часто встречающитхся (их 138 штук)
print(repairs_type_filter)

# Тут данные могут содержать меньше кодов, чем исходный набор ds, поэтому перекодируем повторно.
(ds, code2_dict) = oper_loader_and_encoder.encode(ds, out_code2_path)
voc_size = len(code2_dict) + 1
print(voc_size) #576

# Деление на тестовые и проверочные наборы данных (163136 244706)
(train_x, train_y, test_x, test_y) = oper_preprocessor.split_to_train_and_test(ds, 0.7, True) # train - 70%, test - 30%

# Модель и обучение.

import oper_training_02
(model, history) = oper_training_02.train(train_x, train_y, test_x, test_y, voc_size)


# x = test_x[0:1]
# a = model.predict_classes(x)
# print(a)

def predict(data, predict_len):
    """
    Функция предсказания [N+1, N+2, N+3] элементов последовательности размера N.
    Предсказываем N+1, добавляем этот элемент в конец последовательности, сдвигаем начало окна на 1 элемент вперед.
    Предсказываем следующий элемент сдвинутого окна, находим N+2, добавляем его в конец окна, сдвигаем окно на 1 элемент вперед.
    Предсказываем следующий элемент сдвинутого окнаб находим N+3.
    Список полученных предсказаний [N+1, N+2, N+3] добавляем в y_pred как строку.
    """

    print('Predicting...')
    y_data_predict = [] ## todo матрица двумерная
    for x_index in range(0, len(data), 1): # в диапазоне массивчиков
        x = data[x_index:(x_index+1)]
        print('x', x)
        y = np.zeros(shape=(0,0)) # []
        print('y', y) # []
        for predict_index in range(0, predict_len, 1):
            y_element = model.predict_classes(data[x_index:(x_index+1)])
            print('y_element', y_element)
            y = np.append(y, y_element)
            print('y', y)
            x = np.append(x, y_element)
            x = x[1:]
            data[x_index:(x_index+1)] = x  #для предикта трех разных значений [1 2 3]
        y_pred_three = list(map(int, y))  # массив -> в список integer для сохранения, как строку в матрице
        print('y_end', y_pred_three)

        y_data_predict.append(y_pred_three) ## todo как строка матрицы
        print('y_predict', y_data_predict)

    return (y_data_predict)

y_data_predict = predict(test_x, 3)
print(y_data_predict)
#%%

# Запись массива y_predict в файл
with open('y_predict_for_three_elements.txt', 'w') as filehandle:
    json.dump(y_data_predict, filehandle)

#%%
# Считывание массива y_predict из файла
with open('y_predict_for_three_elements.txt', 'r') as filehandle:
    y_data_predict = json.load(filehandle)

print(len(y_data_predict)) # 122353
y_data_predict = list(y_data_predict)

#%%
def accuracy_speical (y_test, y_predict):
    """
    Вероятность того, что y_test равен одному из трех

    Функция для расчета точности предсказания N+1, N+2, N+3 элементов.
    Сравниваем y_test с каждым из предсказанных трех элементов
    на наличие равенства y_test хотя бы одному из y_predict.
    """
    # y_test M_test * 1
    # y_predict M_test * predict_len
    predict_len = 3
    hit = 0
    y_1 = 0
    y_2 = 0
    y_3 = 0
    accuracy_1 = 0
    accuracy_2 = 0
    accuracy_3 = 0
    for y_index in range (0, len(y_test)):
        for i in range(0, predict_len):
            print(y_test[y_index]) # 4
            print(y_predict[y_index][i])
            if y_test[y_index] == y_predict[y_index][i]:
                hit += 1
                print('hit', hit)
        # print('----------------------------------')
        if hit == 1:
            y_1 += 1
        if hit == 2:
            y_2 += 1
        if hit == 3:
            y_3 += 1
        hit = 0
    accuracy_1 += y_1 * 100 / len(test_y) # точность того, что 1 элемент из трех будет равен y_test
    accuracy_2 += y_2 * 100 / len(test_y) # точность того, что 2 элемента из трех будут равны y_test
    accuracy_3 += y_3 * 100 / len(test_y) # точность того, что 3 элемента из трех будут равны y_test

    # print('y_1', y_1)
    # print('y_2', y_2)
    # print('y_3', y_3)
    # print('accuracy_1', accuracy_1)
    # print('accuracy_2', accuracy_2)
    # print('accuracy_3', accuracy_3)
    return accuracy_1, accuracy_2, accuracy_3

accuracy_1, accuracy_2, accuracy_3 = accuracy_speical (test_y, y_data_predict )


#%%
x = [1, 2, 3]
y = [accuracy_1, accuracy_2, accuracy_3]
plt.bar(x, y)
plt.title('Тосность предсказания 1, 2, 3 из 3 операций')
plt.xlabel('Номер операции')
plt.ylabel('Точность предсказания')
plt.show()

#%%

