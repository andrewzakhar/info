from abc import ABC, abstractmethod
import pandas as pd

class reader_base(ABC):

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

    @property
    def well_list(self):
        return self.__well_list

    @well_list.setter
    def well_list(self, well_list):
        self.__well_list = well_list

    '''DataFrame результатов запроса к источнику '''
    __df_source_data: pd.DataFrame = None
    @property
    def df_source_data(self):
        return self.__df_source_data

    @df_source_data.setter
    def df_source_data(self, df_source_data):
        self.__df_source_data = df_source_data

    def __init__(self, conf):
        self.__conf = conf


class reader_bd_base(reader_base):
    ''' Базовый класс ридера из БД '''
    ''' Получить данные из источника '''

    @abstractmethod
    def get_data(self):
        pass

    ''' Заполнить DataFrame результатом запроса к источнику '''
    def df_mf_fill(self, index_num_col=1) -> pd.DataFrame:
        dic = {}
        col_index = 0
        ind = []
        ''' Номер колонки с данными, которые будут использованы для индекса df_source_data '''
        if self.metadata and self.rows:
            for col_name in self.metadata:
                col_data = []
                for col in self.rows:
                    col_data.append(col[col_index])
                    if col_index == index_num_col:
                        ind.append(col[index_num_col])
                dic.update({col_name[0]: col_data})
                col_index += 1

            if index_num_col < 0:
                self.df_source_data = pd.DataFrame(dic)
            else:
                self.df_source_data = pd.DataFrame(dic, index=ind)
        else:
            self.df_source_data = None
        pass

