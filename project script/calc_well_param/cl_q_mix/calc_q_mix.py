import calc_func
import ccalc_pvt
from CPVT.standing import calc_standing
x_gr = 0
y_gr = 1


class Qmix:

    def __init__(self, Pintake, Tintake, Qliq, Fw, class_pvt, z_class, pseudo_class):
        self.Pintake = Pintake
        self.Tintake = Tintake
        self.Qliq = Qliq
        self.Fw = Fw
        self.class_pvt = class_pvt
        self.z_class = z_class
        self.pseudo_class = pseudo_class

    def calc_rs_bo_m3m3(self, Pintake, Tintake, Qliq, Fw, class_pvt, z_class, pseudo_class):

        class_pvt.calc(Pintake, Tintake, Qliq, Fw, z_class, pseudo_class)
        clc_rs_m3m3 = class_pvt.rs_m3m3
        clc_bo_m3m3 = class_pvt.bo_m3m3

        return clc_rs_m3m3, clc_bo_m3m3

    def getFirstPointNo(self, Rpnew, k, array):
        i = 0
        F = True

        while F:
            F = False
            if i < k - 1:
                if Rpnew > array[i]:
                    i = i + 1
                    F = True

        if i == 0:
            i = 1

        gFPNo = i - 1
        return gFPNo

    def getPoint(self, Rpnew, k, array_func1, array_func2):

        """
        getPoint возращается два значения в точках Rpnew (два графика - две точки Rpnew с разными значениями) при помощи линейной
        интерполяции.
        """
        n = 0
        X1 = 0
        X2 = 0
        y1 = 0
        y2 = 0
        # для первого "графика"
        """
        getFirstPointNo предназначена для нахождения n -- индекса в массиве по точкам "X" - четные элементы массива
        чтобы расположить Rpnew между точками  Х1 и Х2 так, чтобы выполнялось условие Х1 < Rpnew < Х2
        """
        n = self.getFirstPointNo(Rpnew, k, array_func1[x_gr])
        X1 = array_func1[x_gr][n]
        y1 = array_func1[y_gr][n]

        if k > 1:
            X2 = array_func1[x_gr][n + 1]
            y2 = array_func1[y_gr][n + 1]
        else:
            X2 = X1
            y2 = y1

        # делаем проверку - если функция ступенчатая то выдаем не интерполированное значение, а значение в предущей точке
        if Rpnew >= X2:
            pb_a = y2
        else:
            pb_a = y1

        pb_a = (y2 - y1) / (X2 - X1) * (Rpnew - X1) + y1

        # для второго "графика"
        n = 0
        X1 = 0
        X2 = 0
        y1 = 0
        y2 = 0
        n = self.getFirstPointNo(Rpnew, k, array_func2[x_gr])  # данная функция предназначена для поиска двух точек на между которами Rpnew
        X1 = array_func2[x_gr][n]
        y1 = array_func2[y_gr][n]

        if k > 1:
            X2 = array_func2[x_gr][n + 1]
            y2 = array_func2[y_gr][n + 1]
        else:
            X2 = X1
            y2 = y1

        # делаем проверку - если функция ступенчатая то выдаем не интерполированное значение, а значение в предущей точке
        if Rpnew >= X2:
            bob = y2
        else:
            bob = y1

        bob = (y2 - y1) / (X2 - X1) * (Rpnew - X1) + y1
        return pb_a, bob

    def calc_q_mix_rc_m3day(self):
        n = 10
        #  промежуточные переменные для сохранения значений этих параметров
        #  эти параметры должны возвращаться с теми значениями, с которыми они пришли
        rp_m3m3_r = self.class_pvt.rp_m3m3
        rsb_m3m3_r = self.class_pvt.rsb_m3m3
        bob_m3m3_r = self.class_pvt.bob_m3m3
        pb_atma_r = self.class_pvt.pb_atma
        rp_m3m3 = self.class_pvt.rp_m3m3

        self.class_pvt.rsb_m3m3 = self.class_pvt.rp_m3m3
        pb_atma = self.class_pvt.pb_atma
        Ksep = self.class_pvt.ksep_fr
        Rs = 0
        Bo = 0
        Rpnew_with_Ksep = 0
        Rpnew_Ksep_1 = 0
        pb_atma_tab = 0
        Rpnew = 0
        GasSol = 1  # передается в функцию mod_after_separation как аргумент
        GasGoesIntoSolution = 1  # проинициализирован в u7_types

        Rs, Bo = self.calc_rs_bo_m3m3(self.Pintake, self.Tintake, self.Qliq, self.Fw, self.class_pvt, self.z_class, self.pseudo_class)

        Rpnew_with_Ksep = rp_m3m3 - (rp_m3m3 - Rs) * Ksep
        Rpnew_Ksep_1 = rp_m3m3 - (rp_m3m3 - Rs)

        Delta = (pb_atma - 1) / n
        i = 0
        Tintake_tres_C = 101.5
        func_array_1 = [[], []]
        func_array_2 = [[], []]
        while i <= 10:
            pb_atma_tab = 1 + Delta * i
            rsb_m3m3_tab, Bo_m3m3_tab = self.calc_rs_bo_m3m3(pb_atma_tab, Tintake_tres_C, self.Qliq, self.Fw, self.class_pvt,
                                                             self.z_class, self.pseudo_class)
            # добавляем элементы в два массива. Они будут представлять из себя два "графика"
            func_array_1[x_gr].append(rsb_m3m3_tab)  # будет представлять себя массив точек X
            func_array_1[y_gr].append(pb_atma_tab)  # будет представлять себя массив точек Y
            func_array_2[x_gr].append(rsb_m3m3_tab)
            func_array_2[y_gr].append(Bo_m3m3_tab)
            i = i + 1

        if GasSol == GasGoesIntoSolution:
            Rpnew = Rpnew_with_Ksep
        else:
            Rpnew = Rpnew_Ksep_1

        if Rpnew < self.class_pvt.rsb_m3m3:
            self.class_pvt.pb_atma, self.class_pvt.bob_m3m3 = self.getPoint(Rpnew, i, func_array_1, func_array_2)
            self.class_pvt.rsb_m3m3 = Rpnew

        self.class_pvt.rp_m3m3 = Rpnew_with_Ksep

        self.class_pvt.calc(self.Pintake, self.Tintake, self.Qliq, self.Fw, self.z_class, self.pseudo_class)

        # возвращаем в "исходное состояние"
        self.class_pvt.rp_m3m3 = rp_m3m3_r
        self.class_pvt.rsb_m3m3 = rsb_m3m3_r
        self.class_pvt.pb_atma = pb_atma_r
        self.class_pvt.bob_m3m3 = bob_m3m3_r
        self.q_mix_rc_m3day = self.class_pvt.q_mix_rc_m3day

        return 0

