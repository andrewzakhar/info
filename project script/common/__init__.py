import os
import sys

# Настроить маршрут к модулям пакета
pkg_name = 'common'.lower()
cCurrentDir = os.path.split(os.path.abspath(''))[1].lower()
if cCurrentDir != pkg_name:
    sys.path.append(os.path.join(os.path.abspath(''), pkg_name))