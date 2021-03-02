import Скрипт.common.myLogger as l
import pandas as pd


def insert_esp_measure(connection, data_to_insert):
    max_date = data_to_insert['dt'].max()
    min_date = data_to_insert['dt'].min()
    well_id = data_to_insert['well_id'].values[0]
    try:
        cursor = connection.cursor()

        # Adjust the batch size to meet your memory and performance requirements
        batch_size = 1000
        query = """
            merge into esp_measure e_m
            using dual on (well_id = :well_id and param_id = :param_id and dt = TO_DATE(:dt, 'YYYY/MM/DD HH24:MI:SS'))
            when matched then 
                update 
                    set e_m.val = :val
            when not matched then 
                insert (well_id, param_id, dt, val) 
                values (:well_id, :param_id, TO_DATE(:dt, 'YYYY/MM/DD HH24:MI:SS'), :val)        
        """
        data = []
        for value in data_to_insert.values:
            data.append(value)
            if len(data) % batch_size == 0:
                cursor.executemany(query, data)
                data = []
        if data:
            cursor.executemany(query, data)
        connection.commit()
        cursor.close()
    except Exception as e:
        connection.rollback()
        l.logger.error(
            f'Module: "{__name__}"       Function: "insert_esp_measure"       min_date = {min_date}, max_date = {max_date}, well_id = {well_id}      Error: = {e}')


def insert_esp_analysis_result(connection, well_id, param_id, dt_borders):
    try:
        data_to_insert = []
        for small_list in dt_borders:
            small_list.insert(0, param_id)
            small_list.insert(0, well_id)
            data_to_insert.append(small_list)

        cursor = connection.cursor()
        query = """
            insert into esp_analysis_result(well_id, param_id, dt_from, dt_to)
            values(:well_id, :param_id, 
                TO_DATE(:dt_from, 'YYYY/MM/DD HH24:MI:SS'), TO_DATE(:dt_to, 'YYYY/MM/DD HH24:MI:SS'))
        """
        cursor.executemany(query, data_to_insert)
        connection.commit()
        cursor.close()
    except Exception as e:
        connection.rollback()
        l.logger.error(
            f'Module: "{__name__}"       Function: "insert_esp_analysis_result"       well_id = {well_id}, param_id = {param_id}     Error: = {e}')


def insert_esp_analysis_result_common(connection, well_id, dt_from, dt_to, data):
    try:
        cursor = connection.cursor()
        data_to_insert = []
        for param_id, value in data.items():
            small_list = [int(well_id), dt_from, dt_to, param_id, value]
            data_to_insert.append(small_list)
        query = """
            insert into esp_analysis_result_common(well_id, dt_from, dt_to, param_id, value)
            values(:well_id, TO_DATE(:dt_from, 'YYYY/MM/DD HH24:MI:SS'), 
                TO_DATE(:dt_to, 'YYYY/MM/DD HH24:MI:SS'), :param_id, :value)
        """
        cursor.executemany(query, data_to_insert, batcherrors=True)
        connection.commit()
        cursor.close()
    except Exception as e:
        connection.rollback()
        print(e)
        l.logger.error(
            f"""Module: "{__name__}"       Function: "insert_esp_analysis_result_common"       well_id = {well_id}
                , dt_from = {dt_from}, dt_to = {dt_to}     Error: = {e}""")
