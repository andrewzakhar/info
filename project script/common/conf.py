from abc import ABC, abstractmethod

import configparser
import os

class conf(ABC):
    ''' Имя конфигурационного файла '''
    __ini_file_name = 'settings.ini'
    @property
    def ini_file_name(self):
        return self.__ini_file_name

    ''' Полный маршрут к конфигурационному файлу с именем '''
    __ini_file_full_path = None

    ''' Объект конфигурации '''
    __config = None
    @property
    def config(self):
        return self.__config

    def __init__(self, ini_file_name, path_ini):
        self.__current_dir = os.getcwd()
        if ini_file_name != None and ini_file_name != '':
            self.__ini_file_name = ini_file_name
        else:
            self.__ini_file_name = self.ini_file_name
        self.path_ini = path_ini
        self.__config = configparser.ConfigParser()
        self.init_conf()
        self.load_conf()

    '''Текущая папка в файловой системе'''
    __current_dir = None
    @property
    def current_dir(self):
        return self.__current_dir

    __path_ini = ''
    ''' Маршрут к конфигурационному файлу, если пусто - в текущей папке '''
    @property
    def path_ini(self):
        return self.__path_ini

    @path_ini.setter
    def path_ini(self, path_ini):
        if path_ini == None or path_ini == '':
            self.__ini_file_full_path = os.path.join(self.__current_dir, self.__ini_file_name)
        else:
            self.__ini_file_full_path = os.path.join(self.__path_ini, self.__ini_file_name)

    def init_conf(self):
        if os.path.exists(self.__ini_file_full_path):
            self.__config.read(self.__ini_file_full_path)

    # абстрактный метод, который будет необходимо переопределять для каждого подкласса
    @abstractmethod
    def load_conf(self):
        pass


class conf_bd(conf):
    __section_name_ora_conn = 'ora_conn_mf'
    __param_name_conn_user = 'conn_user'
    __param_name_conn_pass = 'conn_pass'
    __param_name_conn_tns = 'conn_tns'

    conn_user = None

    @property
    def conn_user(self):
        return self.__conn_user

    conn_pass = None

    @property
    def conn_pass(self):
        return self.__conn_pass

    conn_tns = None

    @property
    def conn_tns(self):
        return self.__conn_tns

    def load_conf(self, section_name = None):
        csec_name = None
        if section_name == None:
            csec_name = self.__section_name_ora_conn
        else:
            csec_name = section_name
            self.__section_name_ora_conn = section_name
        if self.config.has_section(csec_name) and self.config.has_option(
                csec_name, self.__param_name_conn_user):
            self.__conn_user = self.config.get(csec_name, self.__param_name_conn_user)
        if self.config.has_section(csec_name) and self.config.has_option(
                csec_name, self.__param_name_conn_pass):
            self.__conn_pass = self.config.get(csec_name, self.__param_name_conn_pass)
        if self.config.has_section(csec_name) and self.config.has_option(
                csec_name, self.__param_name_conn_tns):
            self.__conn_tns = self.config.get(csec_name, self.__param_name_conn_tns)


# class conf:
#
#     __section_ora = 'ora_conn'
#     __section_localfile = 'local_files'
#
#     ''' Имя секции (строка), которая в данный момент активна (используется) '''
#     __section_active = None
#
#     @property
#     def section_active(self):
#         return self.__section_active
#
#     @section_active.setter
#     def section_active(self, section_active):
#         self.__section_active = section_active
#
#     ora_user_name = None
#     ora_pass = None
#     ora_tns = None
#
#     '''Настройка маршрута к папке локальной БД в виде файлов txt, csv, xls и т.д.'''
#     __localfile_path = None
#     @property
#     def localfile_path(self):
#         return self.__localfile_path
#
#
#     def read_config(self):
#         if not os.path.exists(self.path_ini):
#             self.__section_active = self.__section_localfile
#         else:
#             __config = configparser.ConfigParser()
#             __config.read(self.path_ini)
#             if __config.has_section("main") and __config.has_option("main", "section_active"):
#                 self.__section_active = __config.get("main", "section_active")

# def createConfig(path):
#     """
#     Create a config file
#     """
#     config = configparser.ConfigParser()
#     config.add_section("Settings")
#     config.set("Settings", "font", "Courier")
#     config.set("Settings", "font_size", "10")
#     config.set("Settings", "font_style", "Normal")
#     config.set("Settings", "font_info",
#                "You are using %(font)s at %(font_size)s pt")
#
#     with open(path, "w") as config_file:
#         config.write(config_file)
#
# def crudConfig(path):
#     """
#     Create, read, update, delete config
#     """
#     if not os.path.exists(path):
#         createConfig(path)
#
#     config = configparser.ConfigParser()
#     config.read(path)
#
#     # Читаем некоторые значения из конфиг. файла.
#     font = config.get("Settings", "font")
#     font_size = config.get("Settings", "font_size")
#
#     # Меняем значения из конфиг. файла.
#     config.set("Settings", "font_size", "12")
#
#     # Удаляем значение из конфиг. файла.
#     config.remove_option("Settings", "font_style")
#
#     # Вносим изменения в конфиг. файл.
#     with open(path, "w") as config_file:
#         config.write(config_file)

# if __name__ == "__main__":
#     path = "settings.ini"
#     createConfig(path)

# co = conf(None)
# co.read_config()
# print(co.section_active)