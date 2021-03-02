from Скрипт.pkv import tools
from Скрипт.pkv.db_writers import common_writer as cv
import pandas as pd
import cx_Oracle

# инициализируем вспомогательный класс
gn = tools.GlobalNames()

filePath = "C:\\work_dir\\3__крайнее__17.05.20__Журнал_2020_05_18_00-52-41_-_Куст_100_Скв_6603.csv"
current_file = pd.read_csv(filePath)
connection = cx_Oracle.connect("ERA_NEURO", "ERA_NEURO", "172.17.254.29:1521/LOCAL", encoding="UTF-8")

# забираем айди скважины (пока что не придумал точно как, поэтому хардкодом)
well_id = '{:04d}'.format(tools.get_well_id('3__крайнее__17.05.20__Журнал_2020_05_18_00-52-41_-_Куст_100_Скв_6603'))

columns_to_rename = gn.return_dict_column_to_rename()

current_file = tools.rename_columns_by_dict(current_file, columns_to_rename)


current_file = current_file.filter(items=gn.renamed_parameters()).drop_duplicates(subset=["dt"])

tmp_list = []
for i, v in current_file.iterrows():
    dt = v['dt']
    for param_id, value in v.items():
        if param_id != 'dt':
            tmp_list.append([well_id, param_id, dt, value])

# пишем данные из csv в бд
data_to_insert = pd.DataFrame(tmp_list, columns=['well_id', 'param_id', 'dt', 'value']).dropna(subset=['value'])
cv.insert_esp_measure(connection, data_to_insert)


