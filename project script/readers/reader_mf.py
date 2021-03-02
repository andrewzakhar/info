from abc import ABC, abstractmethod
from readers import reader
from common import db_connector, conf
from calc_well_param import cesp
import pandas as pd

class reader_mf_base(reader.reader_bd_base):
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

    ''' Получить данные из источника '''
    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def get_esp(self, well_id) -> cesp.esp:
        pass

class reader_mf(reader_mf_base):

    __conn = None
    @property
    def conn(self):
        return self.__conn

    def __init__(self, reader_conf: conf.conf_bd, well_list: list):
        self.well_list = well_list
        self.set_conn(reader_conf)
        self.get_data()
        self.df_mf_fill()  # -1

    ''' Настроить соединение '''
    def set_conn(self, conf):
        if conf != None:
            self.__conn = db_connector.get_conn_single(conf.conn_user, conf.conn_pass, conf.conn_tns)

    ''' Получить данные из источника '''
    def get_data(self):
        query_str = '''
            select ass.REGION_NAME, --ДО 
                ass.OIS_WELL_ID, --Id скважины в OIS
                di.LIQUIDRATE CLIQUIDRATE, -- из характеристики дебит жидкости
                di.HEAD CHEAD,       -- из характеристики  напор
                di.POWER CPOWER,      -- из характеристики мощность
                di.EFFICIENCY CEFFICIENCY,   --из характеристики КПД    
                ass.esp_id, 
                ass.WELL_STATE_CODE, 
                ass.WELL_STATE_NAME,  --состояние скважины, наименование OIS
                ass.FUND_GROUP_CODE,  --группа фонда, id группы фонда из ШТР 
                ass.FUND_GROUP_NAME,  --группа фонда, наименование группы фонда из ШТР 
                ass.WELL_TYPE_CODE,     --тип скажины, код OIS 
                ass.WELL_TYPE_NAME,     --тип скважины, наименование OIS 
                ass.SHOP_NAME,             --Цех
                ass.CLUSTER_NAME,         --Куст 
                ass.WELL_NAME,            
                ass.ASSEMBLYID,         --ID последней сборки, заведённой в системе Эра-Мехфонд  и утверждённой
                ass.ASSEMBLYNAME_BOMD,  --Наименование Эра-Мехфонд
                ass.ASSEMBLYCODE_OIS,   --Код типоразмер насоса OIS 
                ass.ASSEMBLYNAME_OIS,   --Наименование типоразмера OIS 
                ass.ESP_SECTION_COUNT,  --количество секций ЭЦН
                ass.ASSEMBLEDATE,       --дата монтажа 
                ass.DISASSEMBLEDATE,     --дата демонтажа
                ass.PUMP_NAME,          --дополнительное поле насос, может вывести наименование насоса в случае отсутствия паспорта - типоразмер из OIS
                ass.ASSEMBLY_HEAD,      -- напор установки, м3
                ass.ASSEMBLY_RATE,      -- Номинальный дебит установки
                ass.ASSEMBLY_PUMPDEPTH, --глубина спуска
                ass.ASSEMBLY_STAGE_COUNT, --Количество ступеней ЭЦН в установке (сумма по всем секциям), шт
                ass.ESP_MAXSTAGE_COUNT,     --Максимальное количество ступеней первой секции ЭЦН по каталогу производителя, шт
                ass.ESP_MINOPTRATE,     --Левая граница первой секции ЭЦН, м3/сут
                ass.ESP_MAXOPTRATE,     --Правая граница первой секции ЭЦН, м3/сут
                ass.ESP_MODEL,          --Модель первой секции ЭЦН
                ass.ESP_MANUFACTURER,   -- Производитель первой секции ЭЦН
                ass.ESP_FREQUENCY,      -- Номинальная частота первой секции ЭЦН, Гц
                ass.MAX_RATE_1_SECTION_FROM_CURVES MAX_RATE, -- Максимальный дебит установки из характеристики
                ass.MAX_HEAD_1_SECTION_FROM_CURVES MAX_HEAD  -- Максимальный напор установки из характеристики
        from BOMD.VIEW_DARS_ASSEMBLY_INFO ass, BOMD.VIEW_OPTIRAMP_ESP_CURVE_DICT di
        where ass.esp_id = di.esp_id
        '''
        where_clause = ''
        wellid_list_str = []
        order_clause = 'order by OIS_WELL_ID, CLIQUIDRATE'
        if self.well_list != None:
            for wellid in self.well_list:
                if isinstance(wellid, int): # Проверить на тип, чтобы не было подставы
                    wellid_list_str.append(str(wellid))

            where_clause = ', '.join(wellid_list_str)
            where_clause = f'and ass.OIS_WELL_ID in ({where_clause})'
        if len(wellid_list_str) > 0:
            query_str = f'{query_str} {where_clause} {order_clause}'
            self.rows, self.metadata = db_connector.ora_execute(self.conn, query_str)
        else:
            self.rows, self.metadata = None, None
        pass

    ''' Заполнить DataFrame результатом запроса к источнику '''
    def df_mf_fill(self) -> pd.DataFrame:
        dic = {}
        col_index = 0
        index_num_col1 = 1
        index_num_col2 = 2
        ind1 = []
        ind2 = []
        ''' Номер колонки с данными, которые будут использованы для индекса df_source_data '''
        if self.metadata and self.rows:
            for col_name in self.metadata:
                col_data = []
                for col in self.rows:
                    col_data.append(col[col_index])
                    if col_index == index_num_col1:
                        ind1.append(col[col_index])
                    if col_index == index_num_col2:
                        ind2.append(col[col_index])
                dic.update({col_name[0]: col_data})
                col_index += 1
            ind_arr = [ind1, ind2]
            ind = pd.MultiIndex.from_arrays(arrays=ind_arr, names=['well_id', 'CLIQUIDRATE'])
            self.df_source_data = pd.DataFrame(dic, index=ind)
        else:
            self.df_source_data = None
        pass


    def get_esp(self, well_id) -> cesp.esp:
        esp_row = self.df_source_data.loc[well_id]
        if len(esp_row.index) > 0:
            _id_pump = esp_row['ASSEMBLYCODE_OIS'][0]
            _manufacturer_name = esp_row['ESP_MANUFACTURER'][0]
            _pump_name=esp_row['PUMP_NAME'][0]
            _freq_hz=esp_row['ESP_FREQUENCY'][0]
            _esp_nom_rate_m3day = esp_row['ASSEMBLY_RATE'][0]
            _esp_max_rate_m3day = esp_row['MAX_RATE'][0]
            # esp_row = esp_row.sort_values(by='CLIQUIDRATE')
            esp_row.sort_index(inplace=True)
            esp_rates = []
            esp_heads = []
            esp_eff = []
            esp_powers = []
            for ind in range(len(esp_row.index)):
                esp_rates.append(esp_row.iloc[ind]['CLIQUIDRATE'])
                esp_heads.append(esp_row.iloc[ind]['CHEAD'])
                esp_powers.append(esp_row.iloc[ind]['CPOWER'])
                esp_eff.append(esp_row.iloc[ind]['CEFFICIENCY'])

            esp_heads_с = cesp.polinom_solver(esp_rates, esp_heads, 5)
            esp_polynom_head = cesp.esp_polynom(esp_heads_с)

            esp_eff_с = cesp.polinom_solver(esp_rates, esp_eff, 5)
            esp_polynom_efficency = cesp.esp_polynom(esp_eff_с)

            esp_powers_с = cesp.polinom_solver(esp_rates, esp_powers, 5)
            esp_polynom_power = cesp.esp_polynom(esp_powers_с)

            new_esp = cesp.esp(id_pump=_id_pump,
                               manufacturer_name=_manufacturer_name, pump_name=_pump_name,
                               freq_hz=_freq_hz, esp_nom_rate_m3day=_esp_nom_rate_m3day, esp_max_rate_m3day=_esp_max_rate_m3day,
                               esp_polynom_head_obj=esp_polynom_head, esp_polynom_efficency_obj=esp_polynom_efficency,
                               esp_polynom_power_obj=esp_polynom_power)
            # esp_max_rate_m3day=esp_row['ESP_MAXOPTRATE'][well_id]
            new_esp.stage_num = esp_row['ASSEMBLY_STAGE_COUNT'][0]
            new_esp.head = esp_row['ASSEMBLY_HEAD'][0]
            new_esp.pump_depth_m = esp_row['ASSEMBLY_PUMPDEPTH'][0]
        return new_esp
