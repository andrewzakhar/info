import cx_Oracle

from Скрипт.pkv import tools
from Скрипт.pkv import analyse_tools as at
from Скрипт.pkv.db_reader import common_reader as cr

# инициализируем вспомогательный класс
gn = tools.GlobalNames()

# условные параметры на вход для анализа
date_start = '14/03/2020 6:55:16'
date_end = '18/06/2020 7:55:16'
well_id = '{:10d}'.format(2860066700)

connection = cx_Oracle.connect("ERA_NEURO", "ERA_NEURO", "172.17.254.29:1521/LOCAL", encoding="UTF-8")

df = cr.get_esp_measure(connection, date_start, date_end, well_id)

qwe, qwe1 = at.analyse_pkv(df, well_id, connection)
#work_mode = tools.get_work_mode(df, well_id, gn.motor_load_perc)