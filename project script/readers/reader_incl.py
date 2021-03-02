from abc import ABC, abstractmethod
from readers import reader
from common import db_connector, conf
import pandas as pd

class reader_incl_base(reader.reader_bd_base):
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

    __depth_angle_df = None
    @property
    def depth_angle_df(self):
        return self.__depth_angle_df

    @depth_angle_df.setter
    def depth_angle_df(self, depth_angle_df):
        self.__depth_angle_df = depth_angle_df

    def __init__(self, well_list: list, conf):
        self.__well_list = list

    @abstractmethod
    def get_depths_list(self, well_id, bore_num=None) -> ():
        '''
        Вернуть список глубин инклиметрии по коду скважины и стволу
        :param well_id: Код скважины
        :param bore_num: Текущий ствол bore_num None, обычно определяется автоматически
        :return:
        '''
        pass

    @abstractmethod
    def get_angle_depth(self, depth) -> ():
        '''
        Вернуть угол по глубине
        :param depth: глубина
        :return: угол, отклонение от горизонтали
        '''
        pass


class reader_incl_ora(reader_incl_base):

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
        ''' В запросе выбрать инклиметрию по одной скважине или нескольким
        '''
        query_str = '''
  SELECT kl.SK_1 well_id,
         kl.MS_1 field_id,
         kl.S1_1 well_name,
         kl.GL_1 Depth,
         kl.UG_1 angle,
         kl.AZ_1 azimuth,
         kl.UL_1 udl,
         kl.ZZ_1,
         kl.XX_1,
         kl.YY_1,
         kl.SM_1,
         kl.DU_1,
         kl.KR_1,
         kl.TK_1,
         kl.GLN_1,
         kl.KR_MAX_1,
         kl.ZB2_1 BORE_NUM,
         kl.GRAZ_1
    FROM klinz kl
        INNER JOIN rabpl rb on rb.sk_1 = kl.sk_1 and rb.d2_1 = 99999999
        INNER JOIN plast pl on pl.sk_1 = kl.sk_1 and pl.zb2_1 = kl.zb2_1 and pl.pl_1 = rb.pl_1 and pl.d2_1 = 99999999     
        '''
        where_clause = "where 1 = 1 "
        where_clause_add = ''
        order_by = 'order by kl.sk_1, kl.gl_1'
        if self.well_list != None:
            wellid_list_str = []
            for wellid in self.well_list:
                if isinstance(wellid, int): # Проверить на тип, чтобы не было подставы
                    wellid_list_str.append(str(wellid))

            where_clause_add = ', '.join(wellid_list_str)
            where_clause = f'{where_clause} and kl.sk_1 in ({where_clause_add})'
        query_str = f'{query_str} {where_clause} {order_by}'
        self.rows, self.metadata = db_connector.ora_execute(self.conn, query_str)
        self.depth_angle_df = None

    def get_depths_list(self, well_id, bore_num=None) -> ():
        rows = None
        if well_id != None:
            if well_id in self.df_source_data.index:
                rows = self.df_source_data.loc[[well_id], ['DEPTH', 'BORE_NUM', 'ANGLE']]
                if bore_num is not None:
                    self.depth_angle_df = rows[rows['BORE_NUM'] == bore_num]
                else:
                    self.depth_angle_df = rows
                if self.depth_angle_df.empty != True:
                    rows = self.depth_angle_df['DEPTH']
                    rows = tuple(rows.values)
        return rows

    def get_angle_depth(self, depth) -> ():
        '''
        Вернуть угол по глубине
        :param depth: глубина
        :return: угол, отклонение от горизонтали
        '''
        angle = None
        row = self.depth_angle_df[self.depth_angle_df['DEPTH'] == depth]
        if row.empty != True:
            angle = 90 - row.iloc[0]['ANGLE']
        return angle
