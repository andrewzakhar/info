from abc import ABC, abstractmethod

class solver(ABC):
    __solver_datas = None
    def __init__(self, solver_datas):
        self.__solver_datas = solver_datas

    # абстрактный метод, который будет необходимо переопределять для каждого подкласса
    @abstractmethod
    def calc(self):
        pass