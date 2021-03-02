from abc import ABC, abstractmethod
from readers import reader
from common import db_connector, conf

class reader_tm_base(reader.reader_bd_base):

    __well_id = None
    ''' Скважина OIS '''
    @property
    def well_id(self):
        return self.__well_id

    @well_id.setter
    def well_id(self, well_id):
        self.__well_id = well_id

    __dt_begin = None
    ''' Дата запроса '''
    @property
    def dt_begin(self):
        return self.__dt_begin

    @dt_begin.setter
    def dt_begin(self, dt_begin):
        self.__dt_begin = dt_begin

    __dt_end = None
    ''' Дата запроса '''
    @property
    def dt_end(self):
        return self.__dt_end

    @dt_end.setter
    def dt_end(self, dt_end):
        self.__dt_end = dt_end

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

    def __init__(self, conf, well_id: int):
        self.__well_id = well_id

    @abstractmethod
    def get_data(self, begin_date, end_date):
        pass

class reader_tm_ora(reader_tm_base):

    __conn = None
    @property
    def conn(self):
        return self.__conn

    def __init__(self, reader_conf: conf.conf_bd, well_id: int):
        '''  Имя секции с настройками подключения в файле настроек '''
        self.section_name = 'ora_conn_wellop'
        self.well_id = well_id
        self.set_conn(reader_conf)

    ''' Настроить соединение '''
    def set_conn(self, conf):
        if conf != None:
            conf.load_conf(self.section_name)
            self.__conn = db_connector.get_conn_single(conf.conn_user, conf.conn_pass, conf.conn_tns)

    ''' Получить данные из источника '''
    def get_data(self, begin_date, end_date):
        ''' В запросе выбрать ТМ по одной скважине
        '''
        self.dt_begin = begin_date
        self.dt_end = end_date
        query_str = '''
        with 
            dta as (select :dt_begin as dt_begin, :dt_end as dt_end from dual)
        select dt_hour, LIQ_RATE, OIL_RATE, WATER_CUT, PLIN, QGAS, FREQ_HZ, ACTIV_POWER, PED_T, PINP 
        from 
            (select * from 
                (select dt_hour, tm.param_id, avg(avg_val) val
                from     
                    (select tm.param_id, trunc(dt, 'HH24') dt_hour, AVG(val) OVER (PARTITION BY param_id ORDER BY dt RANGE BETWEEN INTERVAL '1' HOUR PRECEDING AND CURRENT ROW) avg_val
                    from tm_chess tm, dta 
                    where tm.well_id = :well_id and tm.param_id in (12, 30, 33, 142, 204, 220, 403, 187, 188) 
                        and dt >= dta.dt_begin and dt < dta.dt_end) tm
                group by param_id, dt_hour)
            pivot (avg(val) for param_id in (12 as LIQ_RATE, 30 as OIL_RATE, 33 as WATER_CUT, 142 as PLIN, 204 as QGAS, 220 as FREQ_HZ, 403 as ACTIV_POWER, 187 PED_T, 188 PINP)))
        '''

        where_clause = ""
        where_clause_add = ''
        order_by = 'order by dt_hour'
        params = {}
        if self.well_id != None and self.dt_begin is not None and self.dt_end is not None:
            params = {'well_id': self.well_id, 'dt_begin': self.dt_begin, 'dt_end': self.dt_end}
            query_str = f'{query_str} {where_clause} {order_by}'
            self.rows, self.metadata = db_connector.ora_execute(self.conn, query_str, params)
            self.df_mf_fill(index_num_col=0)
            return self.df_source_data
