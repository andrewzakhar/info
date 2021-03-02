from abc import ABC, abstractmethod
from readers import reader
from common import db_connector, conf
import pandas as pd

class reader_kolon_base(reader.reader_bd_base):
    __well_list = []
    ''' Список скважин OIS '''
    @property
    def well_list(self):
        return self.__well_list

    @well_list.setter
    def well_list(self, well_list):
        self.__well_list = well_list

    __rows = []
    ''' list строки данных для подготовки df '''
    @property
    def rows(self):
        return self.__rows

    @rows.setter
    def rows(self, rows):
        self.__rows = rows

    __metadata = []
    ''' list метаданные (имена колонок) для подготовки df '''
    @property
    def metadata(self):
        return self.__metadata

    @metadata.setter
    def metadata(self, metadata):
        self.__metadata = metadata

    def __init__(self, well_list: list, conf):
        self.__well_list = list

class reader_kolon_ora(reader_kolon_base):

    __conn = None
    @property
    def conn(self):
        return self.__conn

    def __init__(self, reader_conf: conf.conf_bd, well_list: list):
        '''  Имя секции с настройками подключения в файле настроек '''
        self.section_name = 'ora_conn_wellop'
        self.well_list = well_list
        self.set_conn(reader_conf)
        self.get_data()
        self.df_mf_fill(index_num_col=0)

    ''' Настроить соединение '''
    def set_conn(self, conf):
        if conf != None:
            conf.load_conf(self.section_name)
            self.__conn = db_connector.get_conn_single(conf.conn_user, conf.conn_pass, conf.conn_tns)

    ''' Получить данные из источника '''
    def get_data(self):
        ''' В запросе выбрать строку с внутренним диаметром э/к kolon_diam_inner_mm по глубине спуска section_depth
            входное значение глубины <= section_depth
        '''
        query_str = '''
        select sk_1, gk_1 as kolon_depth, gs_1 as step_depth, gc_1 as section_depth, ns_1 as step_num, nc_1 as section_num,
        gln_1 section_depth_top, dk_1 kolon_diam_spr_mm, ds_1 kolon_diam_inner_mm 
        from kolon 
        '''
        where_clause = "where YQ_1 = 'YQ0040' "
        where_clause_add = ''
        order_by = 'order by sk_1, gk_1, gs_1, gc_1'
        if self.well_list != None:
            wellid_list_str = []
            for wellid in self.well_list:
                if isinstance(wellid, int): # Проверить на тип, чтобы не было подставы
                    wellid_list_str.append(str(wellid))

            where_clause_add = ', '.join(wellid_list_str)
            where_clause = f'{where_clause} and sk_1 in ({where_clause_add})'
        query_str = f'{query_str} {where_clause} {order_by}'
        self.rows, self.metadata = db_connector.ora_execute(self.conn, query_str)
        pass


    ''' Получить диаметр экспл.колонны по скважине и глубине спуска '''
    def get_ekolon_row(self, well_id, depth):
        kolon_diam_inner_mm = None
        if depth != None and well_id != None:
            if well_id in self.df_source_data.index:
                rows = self.df_source_data.loc[[well_id], ['STEP_DEPTH', 'SECTION_DEPTH', 'KOLON_DIAM_INNER_MM']]
                rows = rows[rows['SECTION_DEPTH'] > depth]
                if rows.empty != True:
                    kolon_diam_inner_mm = rows.iloc[0]['KOLON_DIAM_INNER_MM']
        return kolon_diam_inner_mm
