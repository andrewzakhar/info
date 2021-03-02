from abc import ABC, abstractmethod
import solver

class solver_esp(solver):
    __solver_data = None
    def __init__(self, solver_data):
        self.__solver_data = solver_data

    # абстрактный метод, который будет необходимо переопределять для каждого подкласса
    def calc(self):
        pass