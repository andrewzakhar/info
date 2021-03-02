from abc import ABC, abstractmethod
import pandas as pd

class writer_base(ABC):

    __section_name = None
    '''  Имя секции с настройками подключения в файле настроек '''
    @property
    def section_name(self):
        return self.__section_name

    @section_name.setter
    def section_name(self, section_name):
        self.__section_name = section_name

    __conf = None
    ''' объект конфигурации '''
    @property
    def conf(self):
        return self.__conf
    @conf.setter
    def conf(self, conf):
        self.__conf = conf

    '''DataFrame с данными для сохранения '''
    __df_source_data: pd.DataFrame = None
    @property
    def df_source_data(self):
        return self.__df_source_data

    @df_source_data.setter
    def df_source_data(self, df_source_data):
        self.__df_source_data = df_source_data

    @abstractmethod
    def write_data_df(self, data_frame: pd.DataFrame = None):
        self.df_source_data = data_frame
        pass

    def __init__(self, conf):
        self.conf = conf


