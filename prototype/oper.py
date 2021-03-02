#%%
import numpy as np
from tensorflow.keras.models import load_model
import oper_loader_and_encoder
import oper_preprocessor
import oper_predict

out_opers_code_path = 'data/out_opers_code.csv'
out_params_code_path = 'data/out_params_code.csv'
out_params_values_code_path = 'data/out_params_values_code.csv'
out_repairs_code_path = 'data/out_repairs_code.csv'
out_code2_path = 'data/out_code2.csv'
out_opers_with_accuracy_path = 'data/opers_with_accuracy.csv'
out_count_last_opers_path = 'data/count_last_opers.csv'

param_count = 4 # количество задаваемых параметров, т.е. если 4, то 2 параметра + 2 значения параметров
seq_len = param_count + 3 # 3 - тип ремонта + дата + id_oper

repairs_windowed_dataset_path = 'data/out_repairs_windowed' + str(seq_len) + '.npy'

# Создание массива идентификаторов параметров для дальнешейго вызова в функции
param_arr = oper_loader_and_encoder.param_arr('data/2020-06-19(41_repairs)_dump_opers_parms_v.csv')

def prepare_data():
    print('Loading...')
    # Получение набора данных repairs
    repairs, opers_dict, params_dict, params_values_dict = oper_loader_and_encoder.load_svod(
        'data/2020-06-19(41_repairs)_dump_opers_v.csv',
        'data/2020-06-19(41_repairs)_dump_opers_parms_v.csv',
        'data/2020-06-19(41_repairs)_dump_repairs_v.csv',
        True,
        param_arr, # массив загружаемых параметров
        param_count, # количество параметров
        False,
        out_opers_code_path,
        out_params_code_path,
        out_params_values_code_path,
        out_repairs_code_path)

    # Сохранение загруженного и закодированного с помощью метода load_svod массива данных (repairs) в файл (data/out_repairs.csv).
    oper_loader_and_encoder.dump_loaded_svod_data(repairs, 'data/out_repairs.csv')
    # repairs = oper_loader.load_svod_data_dump('data/out_repairs.csv')
    repairs = np.array([np.array(r) for r in repairs])
    # Возвращает numpy-массив разрезанных "окнами" данных, где win_size - какое количество операций входит в одно окно
    win_size = 5
    repairs_windowed = oper_preprocessor.get_windowed_data(repairs, win_size, param_count)
    np.save(repairs_windowed_dataset_path, repairs_windowed)

    return repairs, opers_dict, params_dict, params_values_dict

# Возврат ремонтов и словарей c операциями, параметрами, значениями параметров
repairs,  opers_dict, params_dict, params_values_dict = prepare_data()
opers_voc, params_voc, params_values_voc = len(opers_dict), len(params_dict), len(params_values_dict)

# Загружаем датасет
ds = np.load(repairs_windowed_dataset_path)
print('ds', ds) # то же самое, что и repairs_windowed

#%%
# Типы ремонтов, по которым есть более указанного числа последовательностей min_m.
repairs_type_filter = oper_preprocessor.repairs_types_filter(ds, 20)

#%%
# Деление на тестовые и проверочные наборы данных по дате
(train_x, train_y, test_x, test_y) = oper_preprocessor.split_to_train_and_test_with_date(ds, 202005)

#%%
# Модель и обучение.
import oper_training_02

# Разделим разрезанные окнами данные на операции, параметрыи и значения параметров для подачи каждого в сеть, как отдельный вход
oper_input, param_input, param_value_input = oper_preprocessor.get_net_inputs(train_x, param_count)
oper_test, param_test, param_value_test = oper_preprocessor.get_net_inputs(test_x, param_count)

# Модель и обучение
model = oper_training_02.net_with_three_inputs(train_x, train_y, test_x, test_y,
                                               opers_voc, params_voc, params_values_voc,
                                               oper_input, param_input, param_value_input,
                                               oper_test, param_test, param_value_test)
#%%