from abc import ABC, abstractmethod
from readers import reader
from common import db_connector, conf
from pvt.cpvt import pvt
from calc_well_param import const
from datetime import datetime

class reader_pvt_base(reader.reader_bd_base):

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

    __dt = None
    ''' Дата запроса '''
    @property
    def dt(self):
        return self.__dt

    @dt.setter
    def dt(self, dt):
        self.__dt = dt

    def __init__(self, reader_conf: conf.conf_bd, well_list: list, dt: datetime):
        '''  Имя секции с настройками подключения в файле настроек '''
        self.well_list = well_list
        self.set_conn(reader_conf)
        self.get_data()
        self.df_mf_fill(index_num_col=0)
        self.dt = dt

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_pvt(self, well_id) -> pvt:
        pass


class reader_pvt_ora(reader_pvt_base):

    __conn = None
    @property
    def conn(self):
        return self.__conn

    def __init__(self, reader_conf: conf.conf_bd, well_list: list, dt: datetime):
        '''  Имя секции с настройками подключения в файле настроек '''
        self.section_name = 'ora_conn_wellop'
        self.well_list = well_list
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
        ''' В запросе выбрать pvt по одной скважине или нескольким
        '''
        query_str = '''
        with 
            dta as (select :dt_begin as dt_begin, :dt_end as dt_end, 3 round_val from dual)
        select well_id, plast_names, round_val, FIELD_CODE, FIELD_NAME, WELL_NAME,
                round(GAMMA_GAS_SK, round_val) GAMMA_GAS_SK, round(GAMMA_GAS_SPL, round_val) GAMMA_GAS_SPL, 
                round(GAMMA_OIL_SK, round_val) GAMMA_OIL_SK, round(GAMMA_OIL_SPL, round_val) GAMMA_OIL_SPL,  
                round(GAMMA_WAT_SK, round_val) GAMMA_WAT_SK, round(GAMMA_WAT_SPL, round_val) GAMMA_WAT_SPL,   
                round(PB_ATMA_SK, round_val) PB_ATMA_SK, round(PB_ATMA_SPL, round_val) PB_ATMA_SPL,  round(TRES_C_SK, round_val) TRES_C_SK, round(TRES_C_SPL, round_val) TRES_C_SPL, 
                round(BOB_M3M3_SK, round_val) BOB_M3M3_SK, round(BOB_M3M3_SPL, round_val) BOB_M3M3_SPL, round(MUOB_CP_SK, round_val) MUOB_CP_SK, round(MUOB_CP_SPL, round_val) MUOB_CP_SPL, 
                round(RSB_M3M3_SK, round_val)RSB_M3M3_SK, round(RSB_M3M3_SPL, round_val) RSB_M3M3_SPL, round(RP_M3M3_SK, round_val) RP_M3M3_SK
        from         
            (select rb.sk_1 well_id, listagg(cl.ns_1, ', ') within group (order by cl.ns_1) plast_names, max(dta.round_val) round_val, 
                max(rb.ms_1) FIELD_CODE, max(cl2.ne_1) FIELD_NAME, trim(max(spl.s1_1)) WELL_NAME, 
                sum(spl.pg_1 * skpl.rp_1 / 100) GAMMA_GAS_SK, sum(sppl.pg_1 * skpl.rp_1 / 100) GAMMA_GAS_SPL,  
                sum(spl.hs_1 / spl.pv_1 * skpl.pn_1 / 100) GAMMA_OIL_SK, sum(sppl.hs_1 / sppl.pv_1 * skpl.pn_1 / 100) GAMMA_OIL_SPL,
                sum(spl.pv_1 * skpl.pv_1 / 100) GAMMA_WAT_SK, sum(SPPL.PV_1 * skpl.pv_1 / 100) GAMMA_WAT_SPL,  
                sum(spl.DN_1 * skpl.rp_1 / 100) PB_ATMA_SK, sum(sppl.dn_1 * skpl.rp_1 / 100) PB_ATMA_SPL,
                sum(spl.TL_1 * skpl.pn_1 / 100) TRES_C_SK, sum(SPPL.TL_1 * skpl.pn_1 / 100) TRES_C_SPL,
                sum(spl.ok_1 * skpl.pn_1 / 100) BOB_M3M3_SK, sum(sppl.ok_1 * skpl.pn_1 / 100) BOB_M3M3_SPL,
                sum(spl.vn_1 * skpl.pn_1 / 100) MUOB_CP_SK, sum(sppl.vn_1 * skpl.pn_1 / 100) MUOB_CP_SPL,
                sum(spl.gf_1 * skpl.rp_1 / 100) RSB_M3M3_SK, sum(sppl.gf_1 * skpl.rp_1 / 100) RSB_M3M3_SPL,
                sum(spl.fo_1 * skpl.rp_1 / 100) RP_M3M3_SK
            from sppl_sk spl
                inner join rabpl rb ON spl.sk_1 = rb.sk_1 and rb.pl_1 = SPL.PL_1 
                inner join dta ON rb.dz_1 <= PKG_OIS.DATE2OIS(dta.dt_end) and rb.d2_1 > PKG_OIS.DATE2OIS(dta.dt_begin) 
                    and spl.dz_1 <= PKG_OIS.DATE2OIS(dta.dt_end) and spl.d2_1 > PKG_OIS.DATE2OIS(dta.dt_begin)
                inner join sppl ON sppl.ms_1 = spl.ms_1 and sppl.pl_1 = spl.pl_1 and sppl.zp_1 = 'ZP000000'  
                inner join skpl ON skpl.sk_1 = spl.sk_1 and skpl.pl_1 = SPL.PL_1 and skpl.dz_1 <= PKG_OIS.DATE2OIS(dta.dt_end) and skpl.d2_1 > PKG_OIS.DATE2OIS(dta.dt_begin)
                inner join class cl on spl.pl_1 = cl.cd_1
                inner join class cl2 on spl.ms_1 = cl2.cd_1
        '''

        where_clause = "where 1=1"
        where_clause_add = ''
        order_by = '''
        group by rb.sk_1)
        order by well_id
        '''
        if self.well_list != None:
            wellid_list_str = []
            for wellid in self.well_list:
                if isinstance(wellid, int): # Проверить на тип, чтобы не было подставы
                    wellid_list_str.append(str(wellid))

            where_clause_add = ', '.join(wellid_list_str)
            where_clause = f'{where_clause} and spl.sk_1 in ({where_clause_add})'
        query_str = f'{query_str} {where_clause} {order_by}'
        params = {}
        if self.dt != None:
            params = {'dt_begin': self.dt, 'dt_end': self.dt}
            self.rows, self.metadata = db_connector.ora_execute(self.conn, query_str, params)

    def set_params_by_field(self, _pvt, field_code):
        if _pvt != None and field_code != None and field_code in const.pvt_fields:
            if _pvt.gamma_gas == None or _pvt.gamma_gas == 0:
                _pvt.gamma_gas = const.pvt_fields[field_code]['gamma_gas']
                
            if _pvt.gamma_oil == None or _pvt.gamma_oil == 0:
                _pvt.gamma_oil = const.pvt_fields[field_code]['gamma_oil']
                
            if _pvt.gamma_wat == None or _pvt.gamma_wat == 0:
                _pvt.gamma_wat = const.pvt_fields[field_code]['gamma_wat']
                
            # if _pvt.rsb_m3m3 == None or _pvt.rsb_m3m3 == 0:
            # Для сходимости модели берем из констант, этого параметра нет в OIS
            _pvt.rsb_m3m3 = const.pvt_fields[field_code]['rsb_m3m3']
                
            if _pvt.pb_atma == None or _pvt.pb_atma == 0:
                _pvt.pb_atma = const.pvt_fields[field_code]['pb_atma']
                
            if _pvt.tres_C == None or _pvt.tres_C == 0:
                _pvt.tres_C = const.pvt_fields[field_code]['tres_C']
                                
            if _pvt.bob_m3m3 == None or _pvt.bob_m3m3 == 0:
                _pvt.bob_m3m3 = const.pvt_fields[field_code]['bob_m3m3']
                
            if _pvt.muob_cP == None or _pvt.muob_cP == 0:
                _pvt.muob_cP = const.pvt_fields[field_code]['muob_cP']

            if _pvt.bwSC_m3m3 == None or _pvt.bwSC_m3m3 == 0:
                _pvt.bwSC_m3m3 = const.pvt_fields[field_code]['bwSC_m3m3']
        return _pvt

    def get_pvt(self, well_id) -> pvt:
        rows = None
        _pvt = None
        if well_id != None:
            if well_id in self.df_source_data.index:
                rows = self.df_source_data.loc[[well_id], 
                        ['WELL_ID', 'FIELD_CODE', 'FIELD_NAME', 
                         'GAMMA_GAS_SK', 'GAMMA_OIL_SK', 'GAMMA_WAT_SK', 'PB_ATMA_SK', 'TRES_C_SK', 'BOB_M3M3_SK',
                         'MUOB_CP_SK', 'RSB_M3M3_SK', 'RP_M3M3_SK', 
                         'GAMMA_GAS_SPL', 'GAMMA_OIL_SPL', 'GAMMA_WAT_SPL', 'PB_ATMA_SPL', 'TRES_C_SPL', 'BOB_M3M3_SPL',
                         'MUOB_CP_SPL', 'RSB_M3M3_SPL', 'RP_M3M3_SK'
                         ]]
                if rows.empty != True:
                    _pvt = pvt()
                    _pvt.gamma_gas = rows.iloc[0]['GAMMA_GAS_SK']
                    _pvt.gamma_oil = rows.iloc[0]['GAMMA_OIL_SK']
                    _pvt.gamma_wat = rows.iloc[0]['GAMMA_WAT_SK']
                    _pvt.pb_atma = rows.iloc[0]['PB_ATMA_SK']
                    _pvt.tres_C = rows.iloc[0]['TRES_C_SK']
                    _pvt.bob_m3m3 = rows.iloc[0]['BOB_M3M3_SK']
                    _pvt.muob_cP = rows.iloc[0]['MUOB_CP_SK']
                    _pvt.rsb_m3m3 = rows.iloc[0]['RSB_M3M3_SK']
                    _pvt.rp_m3m3 = rows.iloc[0][20] # 'RP_M3M3_SK'
                    # _pvt.rp_m3m3 = rows.iloc[0]['RP_M3M3_SK'] # 20
                    field_code = rows.iloc[0]['FIELD_CODE']
                    _pvt = self.set_params_by_field(_pvt, field_code)
        return _pvt
