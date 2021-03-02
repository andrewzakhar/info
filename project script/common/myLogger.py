import os

import logging
import logging.config
import yaml

#Фильтр, который будет оставлять только требуемый уровень логов (значение требуемого уровня задается единственным параметром в конструкторе класса, например, logging.ERROR)
class filter_logs_lvl(logging.Filter):
    def __init__(self, logs_level):
        self.logs_level = logs_level

    def filter(self, record):
        return record.levelno == self.logs_level
#Открываем конфиг файл формата .yaml в той же директории, что и главное исполняемое приложение

with open(f'{os.path.dirname(__file__)}/../config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger('sampleLogger')
