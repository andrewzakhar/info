from abc import ABC, abstractmethod
from readers import reader
from common import db_connector, conf
from datetime import datetime


class reader_tr_base(reader.reader_bd_base):

    __well_id = None
    ''' Скважина OIS '''
    @property
    def well_id(self):
        return self.__well_id

    @well_id.setter
    def well_id(self, well_id):
        self.__well_id = well_id

    __dt = None
    ''' Дата запроса '''
    @property
    def dt(self):
        return self.__dt

    @dt.setter
    def dt(self, dt):
        self.__dt = dt

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

    def __init__(self, well_id: int, conf):
        self.__well_id = well_id

    @abstractmethod
    def get_data(self):
        pass

class reader_tr_ora(reader_tr_base):

    __conn = None
    @property
    def conn(self):
        return self.__conn

    def __init__(self, reader_conf: conf.conf_bd, well_id: int, dt: datetime):
        '''  Имя секции с настройками подключения в файле настроек '''
        self.section_name = 'ora_conn_wellop'
        self.well_id = well_id
        self.dt = dt
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
        ''' В запросе выбрать техрежим по одной скважине
        '''
        query_str = '''
        select WELL_ID, CALC_DATE, OIL_RATE, WATER_CUT, QGAS, GAS_FACTOR, SEPARATION_FACTOR, OIL_VISCOSITY, WATER_VISCOSITY
        from
            (select WELL_ID, CALC_DATE, OIL_RATE, WATER_CUT, PG_RATE QGAS, GAS_FACTOR, SEPARATION_FACTOR, OIL_VISCOSITY, WATER_VISCOSITY,
                first_value(calc_date) over (order by calc_date desc) last_calc_date  
            from well_layer_op 
            where well_id = :well_id
                and calc_date <= :dt) 
        where CALC_DATE = last_calc_date
        '''

        where_clause = ""
        where_clause_add = ''
        order_by = 'order by calc_date desc'
        params = {}
        if self.well_id != None and self.dt != None:
            params = {'well_id': self.well_id, 'dt': self.dt}
            query_str = f'{query_str} {where_clause} {order_by}'
            self.rows, self.metadata = db_connector.ora_execute(self.conn, query_str, params)
