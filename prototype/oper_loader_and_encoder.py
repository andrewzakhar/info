#
# Загрузка данных из файлов исходных данных (или выгрузок из базы) и упаковка их с кодированием.
#
# %%

import numpy as np
import random
import xlsxwriter
import dateparser


def encode(repairs, code_dict_path):
    code_dict = dict()
    next_code = 1
    for row_ind in range(repairs.shape[0]):
        for col_ind in range(repairs.shape[1]):
            if col_ind != 1: # пропуск столбца с датой для исключения кодировки
                val = repairs[row_ind, col_ind]
                if val not in code_dict:
                    code_dict[val] = next_code
                    next_code += 1
                code = code_dict[val]
                repairs[row_ind, col_ind] = code

    save_dict_as_csv(code_dict_path, code_dict)
    return (repairs, code_dict)


def save_dict_as_csv(path, dictionary):
    """
    Сохранение словаря в CSV.
    """
    with open(path, 'w') as file:
        for i in dictionary:
            file.write(str(i) + ',' + str(dictionary[i]) + '\n')


def dump_loaded_svod_data(data, path):
    """
    Сохранение загруженного и закодированного с помощью метода load_svod массива данных в файл.
    """
    with open(path, 'w') as file:
        for row in data:
            file.write(','.join([str(i) for i in row]) + '\n')


def load_svod_data_dump(path):
    """
    Загрузка из файла ранее загруженного и закодированного с помощью метода load_svod массива данных.
    """
    data = []
    with open(path, 'r') as file:
        for line in file.readlines():
            items = line.split(',')
            data.append([int(i) for i in items])
    return data


def load_svod(dump_svod_opers_path, dump_svod_opers_parms_path, dump_repairs_path,
              aggregate_to_first_one,
              params_to_load,
              param_count,
              encode_params_values,
              opers_code_dict_path, params_code_dict_path, params_values_code_dict_path, repairs_type_code_dict_path):
    """
    Загрузка сводки по ремонтам.
    :param aggregate_to_first_one загружать только первую операцию в последовательности поряд идущих одинаковых
    :param params_to_load: массив идентификаторов параметров, которые следует включить в выборку.
    :param_count: количество загружаемых параметров для одной операции
    :param encode_params_values: следует ли кодировать параметры (или пишем значения без изменений).
    :return: [ тип_ремонта_код операция_код параметр_код значение_параметра ... ]
        тип_ремонта один раз
    """

    next_code = 1

    # Загружаем и кодируем ремонты.

    # r_id -> r_type
    repairs_type = dict()

    # r_type_id -> r_type_code
    repairs_dict = dict()

    for line in open(dump_repairs_path, 'r').readlines():
        cols = line.split(',')
        r_id = int(cols[0])
        r_type = int(cols[1])
        repairs_type[r_id] = r_type

        if r_type not in repairs_dict:
            repairs_dict[r_type] = next_code
            next_code += 1

    save_dict_as_csv(repairs_type_code_dict_path, repairs_dict)

    # Загружаем и кодируем параметры.

    # p_id -> p_code
    params_dict = dict()

    # p_value -> p_value_code
    params_values_dict = dict()

    # so_id -> p_id -> p_value (float)
    params_values = dict()

    for line in open(dump_svod_opers_parms_path, 'r').readlines():
        ind1 = line.find(',')
        so_id = int(line[:ind1])
        ind2 = line.find(',', ind1 + 1)
        p_id = int(line[ind1 + 1:ind2])
        ind3 = line[ind2 + 1:] # все, что после второй запятой

        # Если параметр не указан, как загружаемый, то игнорируем его.
        if p_id not in params_to_load:
            continue

        if p_id not in params_dict:
            params_dict[p_id] = next_code # кодируем параметр - p_id
            next_code += 1

        # Если есть тире в значении параметра, то игнорировать
        if '-' in ind3:
            continue
        elif line[ind2 + 1] == '"':
            # -2, т.к. последний символ \n (кавычка - предпоследний).
            p_value = float(line[ind2 + 2:-2].replace(',', '.'))
        else:
            p_value = float(line[ind2 + 1:]) # p_value

        if p_value not in params_values_dict: # кодируем значение параметра - p_value
            params_values_dict[p_value] = next_code
            next_code += 1

        if encode_params_values: # если кодируем, то значение p_value становится закодированным
            p_value = params_values_dict[p_value]

        if so_id not in params_values:
            params_values[so_id] = [params_dict[p_id], p_value] # {so_id: [p_code, p_value]}
        else:
            params_values[so_id].append(params_dict[p_id])
            params_values[so_id].append(p_value)

    save_dict_as_csv(params_code_dict_path, params_dict) # закодированные p_id
    save_dict_as_csv(params_values_code_dict_path, params_values_dict) # закодированные p_values

    # Загружаем и кодируем операции и собираем набор данных.

    # o_id -> o_code
    opers_dict = dict()

    data = []
    repair_row = []
    prev_r_id = -1
    prev_o_id = -1
    next_code = 10000 # задаем такое число, чтобы отличать его от параметров

    for line in open(dump_svod_opers_path, 'r').readlines():
        cols = line.split(',')
        so_id = int(cols[0])
        o_id = int(cols[2])
        r_id = int(cols[3])
        # Обработка столбца с датой в формате год+месяц: 20203
        # --------------------------------------------------------
        date = dateparser.parse(cols[4], date_formats=['%d.%m.%Y %H:%M\n'])
        r_date = int(str(date.year) + '' + '{:02d}'.format(date.month))
        # --------------------------------------------------------
        if o_id not in opers_dict:
            opers_dict[o_id] = next_code
            next_code += 1

        if prev_r_id != r_id:
            if len(repair_row) > 0:
                data.append(repair_row)
            r_type = repairs_type[r_id]
            repair_row = [repairs_dict[r_type], r_date]
            prev_r_id = r_id

        if aggregate_to_first_one and prev_o_id != o_id:
            # Если дата из операции меньше текущей даты на ремонте, то берём дату из операции
            if repair_row[1] > r_date:
                repair_row[1] = r_date
            repair_row.append(opers_dict[o_id])

            # Добавление параметров и значений параметров операций
            param_arr = []
            if so_id in params_values:
                for p in params_values[so_id]:
                    param_arr.append(p)

                # Случай, когда количество параметров операции меньше заданного числа
                if len(param_arr) < param_count:
                    # Заполение имеющихся параметров
                    for i in range(len(param_arr)):
                        repair_row.append(param_arr[i])
                    # Заполнение недостающего количества параметров нулями
                    for i in range(len(param_arr) + 1, param_count + 1):
                        repair_row.append(float(0))
                # Случай, когда когда количество параметров больше заданного числа
                else:
                    for i in range(param_count):
                        repair_row.append(param_arr[i])

            else:
                # Случай, когда у операции нет параметров - заполняем полностью нулями в заданным количеством раз
                for i in range(param_count):
                    repair_row.append(float(0))

        prev_o_id = o_id

    if len(repair_row) > 0:
        data.append(repair_row)

    save_dict_as_csv(opers_code_dict_path, opers_dict)

    return data, opers_dict, params_dict, params_values_dict

# Создание массива идентификаторов параметров для дальнешейго вызова в функции
def param_arr(dump_param_path):
    param_arr = []
    for line in open(dump_param_path, 'r').readlines():
        cols = line.split(',')
        param_id = int(cols[1])
        param_arr.append(param_id)
    return param_arr


def code_to_real_oper(code2_dict, dict_with_data):
    '''
    Функция для создания словаря с реальным идентификатором операции и значением
    :param code2_dict: cловарь, в котором ключи - реальные идентификаторы операций {real_id : code_id}
    :param dict_with_data: словарь, в котором значения по ключу - значения точности  {code_id : accuracy)
    :param oper_codes: словарь, {real_id : value)
    :return: oper_codes
    '''

    # поменяем местами ключ и значение по ключу: code2dict[value] = key
    dict2code = {value: key for key, value in code2_dict.items()}
    oper_codes = {}
    for key, value in dict2code.items():
        if key in dict_with_data:
            oper_codes[dict2code.get(key)] = dict_with_data[key]
    return oper_codes


# %%
def create_table_excel(dict_with_list, dict_with_count):
    # Шапка таблицы
    workbook = xlsxwriter.Workbook('Точность предсказания операции из N.xlsx')
    worksheet = workbook.add_worksheet('Лист 1')
    worksheet.write('B2', 'Код закодировванной операции')
    worksheet.write('C2', 'Код реальной операции')
    worksheet.write('D2', 'Наименование')
    worksheet.write('E2', 'Количество')
    worksheet.write('F2', 'N = 1')
    worksheet.write('G2', 'N = 2')
    worksheet.write('H2', 'N = 3')

    # Заполнение данными
    row = 1
    for key in dict_with_list.keys():
        row += 1
        col = 1
        worksheet.write(row, col, key)
        # Запись количества
        worksheet.write(row, col + 3, dict_with_count[key])
        # Запись N точностей
        for item in dict_with_list[key]:
            worksheet.write(row, col + 4, item)
            col += 1
    workbook.close()


# %%
def quicksort(nums):
    if len(nums) <= 1:
        return nums
    else:
        q = random.choice(nums)
        s_nums = []
        m_nums = []
        e_nums = []
        for n in nums:
            if n < q:
                s_nums.append(n)
            elif n > q:
                m_nums.append(n)
            else:
                e_nums.append(n)
        return quicksort(s_nums) + e_nums + quicksort(m_nums)
