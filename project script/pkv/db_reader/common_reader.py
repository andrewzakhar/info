import Скрипт.common.myLogger as l
import pandas as pd

def get_esp_measure(connection, min_date, max_date, well_id):
    try:
        cursor = connection.cursor()
        query = """
            select param_id, to_char( dt, 'YYYY/MM/DD HH24:MI:SS' ), val
            from esp_measure
            where well_id = {}
                and dt >= TO_DATE('{}', 'DD/MM/YYYY HH24:MI:SS')
                and dt <= TO_DATE('{}', 'DD/MM/YYYY HH24:MI:SS')""".format(well_id, min_date, max_date)
        return pd.read_sql(query, connection).rename(columns={"PARAM_ID": "param_id", "TO_CHAR(DT,'YYYY/MM/DDHH24:MI:SS')": "dt", "VAL": "val"})
    except Exception as e:
        print(e)
        l.logger.error(
            f'Module: "{__name__}"       Function: "get_esp_measure"       min_date = {min_date}, max_date = {max_date}, well_id = {well_id}      Error: = {e}')
