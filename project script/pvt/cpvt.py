from abc import ABC, abstractmethod

class pvt(ABC):

    __gamma_gas = None
    ''' плотность газа удельная '''
    @property
    def gamma_gas(self):
        return self.__gamma_gas
    
    @gamma_gas.setter
    def gamma_gas(self, gamma_gas):
        self.__gamma_gas = gamma_gas

    __gamma_oil = None
    ''' плотность нефти удельная '''
    @property
    def gamma_oil(self):
        return self.__gamma_oil
    
    @gamma_oil.setter
    def gamma_oil(self, gamma_oil):
        self.__gamma_oil = gamma_oil
    
    __gamma_wat = None
    ''' плотность воды удельная '''
    @property
    def gamma_wat(self):
        return self.__gamma_wat
    
    @gamma_wat.setter
    def gamma_wat(self, gamma_wat):
        self.__gamma_wat = gamma_wat
        if self.bwSC_m3m3 is None:
            self.bwSC_m3m3 = gamma_wat / 1

    __rsb_m3m3 = None
    ''' газосодержание при давлении насыщения '''

    @property
    def rsb_m3m3(self):
        return self.__rsb_m3m3

    @rsb_m3m3.setter
    def rsb_m3m3(self, rsb_m3m3):
        self.__rsb_m3m3 = rsb_m3m3

    __rp_m3m3 = None
    ''' Газовый фактор добычной в стандартных условиях '''
    @property
    def rp_m3m3(self):
        return self.__rp_m3m3

    @rp_m3m3.setter
    def rp_m3m3(self, rp_m3m3):
        self.__rp_m3m3 = rp_m3m3

    __pb_atma = None
    ''' давление насыщения  (калибровочное значение) '''
    @property
    def pb_atma(self):
        return self.__pb_atma
    
    @pb_atma.setter
    def pb_atma(self, pb_atma):
        self.__pb_atma = pb_atma

    __tres_C = None
    ''' температура пласта, C '''
    @property
    def tres_C(self):
        return self.__tres_C

    @tres_C.setter
    def tres_C(self, tres_C):
        self.__tres_C = tres_C

    __bob_m3m3 = None
    ''' объемный коэффициент при давлении насыщения '''
    @property
    def bob_m3m3(self):
        return self.__bob_m3m3

    @bob_m3m3.setter
    def bob_m3m3(self, bob_m3m3):
        self.__bob_m3m3 = bob_m3m3

    __muob_cP = None
    ''' вязкость нефти при давлении насыщения (калибровочное значение) '''
    @property
    def muob_cP(self):
        return self.__muob_cP

    @muob_cP.setter
    def muob_cP(self, muob_cP):
        self.__muob_cP = muob_cP

    __bwSC_m3m3 = None
    @property
    def bwSC_m3m3(self):
        return self.__bwSC_m3m3

    @bwSC_m3m3.setter
    def bwSC_m3m3(self, bwSC_m3m3):
        if self.__bwSC_m3m3 is None:
            self.__bwSC_m3m3 = bwSC_m3m3

    ''' соленость воды '''
    __salinity_ppm = None
    @property
    def salinity_ppm(self):
        return self.__salinity_ppm

    @salinity_ppm.setter
    def salinity_ppm(self, salinity_ppm):
        self.__salinity_ppm = salinity_ppm

