import pandas as pd
import numpy
import writers.writer as writer
import datetime
import common.db_connector as db_connector

class writer_bd_base(writer.writer_base):
    ''' Базовый класс райтера в БД '''
    ''' Записать данные '''

    __conn = None
    @property
    def conn(self):
        return self.__conn
    @conn.setter
    def conn(self, conn):
        self.__conn = conn

    __cursor = None
    @property
    def cursor(self):
        return self.__cursor
    @cursor.setter
    def cursor(self, cursor):
        self.__cursor = cursor

    def __init__(self, conf, conn):
        self.conf = conf
        if not conn:
            self.get_conn(self.conf)
        else:
            self.conn = conn

    ''' Настроить соединение '''
    def get_conn(self, conf=None):
        if conf != None:
            conf.load_conf(self.section_name)
            self.conn = db_connector.get_conn_single(conf.conn_user, conf.conn_pass, conf.conn_tns)
            return self.conn

    def conn_close(self):
        if not self.conn:
            selfconn.close()

    def get_col_names(self):
        return list(self.df_source_data)

    def transact_begin(self):
        if self.conn:
            self.conn.begin()
        self.cursor = self.conn.cursor()

    def sql_statement(self, sql_text, params):
        self.cursor.execute(sql_text, params)

    def get_sql_params_from_df(self, data_frame: pd.DataFrame, table_name):
        self.df_source_data = data_frame
        col_names = self.get_col_names()
        sql_query = None
        if len(col_names) > 0 and len(self.df_source_data.index) > 0 and table_name:
            sql_col_names = ', '.join(col_names)
            sql_params = ':' + ' , :'.join(col_names)
            sql_query = f'''
                insert into {table_name} ({sql_col_names}) VALUES
                ({sql_params})
            '''
        return sql_query, col_names

    ''' строки с хоть одним nan пока не записываем '''
    def write_data_df(self, data_frame: pd.DataFrame, table_name):
        sql_query, col_names = self.get_sql_params_from_df(data_frame, table_name)
        for ind in range(len(data_frame)):
            row_data = {}
            nan_exist = False
            for col_name in col_names:
                if str(data_frame.iloc[ind][col_name]) == 'nan': # Проверить на Nan
                    nan_exist = True
                    break
                else:
                    val = data_frame.iloc[ind][col_name]
                if isinstance(val, numpy.int64):  # Проверить на тип, чтобы не было подставы
                    val = int(val)
                row_data[col_name] = val
            if not nan_exist:
                self.cursor.execute(sql_query, row_data)

    def transact_end(self):
        self.cursor.close()
        if self.conn:
            self.conn.commit()

    def disconnect(self):
        if self.conn:
            self.conn.close()
