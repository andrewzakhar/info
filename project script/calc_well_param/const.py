# Константы
const_convert_m_mm = 1000
const_convert_mm_m = 1 / const_convert_m_mm
minPpipe_atma = 0.9

# Universal gas constant
const_r = 8.31
const_g = 9.81
const_rho_air = 1.2217
const_gamma_w = 1
const_rho_ref = 1000
const_ZNLF_rate = 0.1

''' константы для конверсии единиц давления из atma в MPa '''
const_convert_m3day_bbl = 6.289810569
const_convert_gpm_m3day = 5.450992992
''' (US) gallon per minute '''
const_convert_m3day_gpm = 1 / const_convert_gpm_m3day
const_convert_m3m3_scfbbl = 5.614583544
const_convert_scfbbl_m3m3 = 1 / const_convert_m3m3_scfbbl
const_convert_bbl_m3day = 1 / const_convert_m3day_bbl
const_conver_day_sec = 86400
const_convert_hr_sec = 3600
const_convert_m3day_m3sec = 1 / const_conver_day_sec

''' константа для конверсии единиц объемных расходов из м3/сут в баррели '''
const_conver_sec_day = 1 / const_conver_day_sec
const_convert_atma_psi = 14.7
const_convert_psi_atma = 1 / const_convert_atma_psi
const_convert_ft_m = 0.3048
const_convert_m_ft = 1 / const_convert_ft_m
const_convert_m_mm = 1000
const_convert_mm_m = 1 / const_convert_m_mm
const_convert_cP_Pasec = 1 / 1000
const_convert_HP_W = 745.69987
'''  метрическая лошадиная сила. следует учесть, что иногда может применяться механическя лошадиная сила (1.013 метрической) '''
const_convert_W_HP = 1 / const_convert_HP_W
const_convert_Nm_dynescm = 1000
const_convert_lbmft3_kgm3 = 16.01846
const_convert_kgm3_lbmft3 = 1 / const_convert_lbmft3_kgm3

''' набор констант для общих ограничений значений переменных '''

const_gamma_gas_min = 0.5 #плотность метана 0.59 - предполагаем легче газов не будет
const_gamma_gas_max = 2     # плотность углеводородных газов (гексан) может доходить до 4, но мы считаем что в смеси таких не много должно быть
const_gamma_gas_default = 0.8    # Значение по умолчанию
const_gamma_water_min = 0.9 # плотность воды от 0.9 до 1.5
const_gamma_water_max = 1.5
const_gamma_oil_min = 0.5   # плотность нефти
const_gamma_oil_max = 1.5
const_P_MPa_min = 0
const_P_MPa_max = 50
const_Salinity_ppm_min = 0
const_Salinity_ppm_max = 265000  # equal to weigh percent salinity 26.5%.  Ограничение по границам применимости корреляций МакКейна
const_rsb_m3m3_min = 0
const_rsb_m3m3_max = 100000 # rsb more that 100 000 not allowed
const_Ppr_min = 0.002
const_Ppr_max = 30
const_Tpr_min = 0.7
const_Tpr_max = 3
const_Z_min = 0.05
const_Z_max = 5

''' Константы PVT по умолчанию '''

# Вызов pvt_fields['MS0255']['gamma_gas']
dic = {}
dic['gamma_gas'] = 0.754
dic['gamma_oil'] = 0.808
dic['gamma_wat'] = 1
dic['rsb_m3m3'] = 15.6
dic['rp_m3m3'] = 186
dic['pb_atma'] = 138.7
dic['tres_C'] = 76
dic['bob_m3m3'] = 1.486
dic['muob_cP'] = 2.846

pvt_fields = {}
pvt_fields['MS0254'] = dic # MS0254	Суторминское

dic = {}
# Из БД
dic['gamma_gas'] = 0.796 # sppl.pg
dic['gamma_oil'] = 0.82 # т/м3  sppl.hs
dic['gamma_wat'] = 1    # т/м3  sppl.pv
dic['rsb_m3m3'] = 15.6  # sppl.gf_1
dic['rp_m3m3'] = 124 # sppl_sk.fo   Газовый фактор общий, считать на лету или брать из ТР
dic['pb_atma'] = 139 # атм  sppl.dn
dic['tres_C'] = 89  # град C sppl.tl
dic['bob_m3m3'] = 1.278 # sppl.ok Объем.коэфф.нефти, м3/ст.м3, ступ.сепар.
dic['muob_cP'] = 7.76 # sppl.vn

# Константы
# dic['gamma_gas'] = 0.796 # sppl.pg
# dic['gamma_oil'] = 0.82 # т/м3  sppl.hs
# dic['gamma_wat'] = 1    # т/м3  sppl.pv
# dic['rsb_m3m3'] = 15.6  # sppl.gf_1
# dic['rp_m3m3'] = 124 # sppl_sk.fo   Газовый фактор общий, считать на лету или брать из ТР
# dic['pb_atma'] = 139 # атм  sppl.dn
# dic['tres_C'] = 89  # град C sppl.tl
# dic['bob_m3m3'] = 1.278 # sppl.ok Объем.коэфф.нефти, м3/ст.м3, ступ.сепар.
# dic['muob_cP'] = 7.76 # sppl.vn

pvt_fields['MS0286'] = dic # MS0286	Вынгаяхинское

dic = {}
dic['gamma_gas'] = 0.712
dic['gamma_oil'] = 0.846
dic['gamma_wat'] = 1
dic['rsb_m3m3'] = 15.6
dic['rp_m3m3'] = 73
dic['pb_atma'] = 122.4
dic['tres_C'] = 85
dic['bob_m3m3'] = 1.168
dic['muob_cP'] = 12.96

pvt_fields['MS0255'] = dic # MS0255	Восточно-Пякутинское

dic = {}
dic['gamma_gas'] = 0.712
dic['gamma_oil'] = 0.853
dic['gamma_wat'] = 1.009
dic['rsb_m3m3'] = 15.6
dic['rp_m3m3'] = 73
dic['pb_atma'] = 119.9
dic['tres_C'] = 87
dic['bob_m3m3'] = 1.168
dic['muob_cP'] = 12.96

pvt_fields['MS0611'] = dic # MS0611	Сугмутское

dic = {}
dic['gamma_gas'] = 0.712
dic['gamma_oil'] = 0.86
dic['gamma_wat'] = 1.009
dic['rsb_m3m3'] = 62 # Это значение из модели
dic['rp_m3m3'] = 53.2
dic['pb_atma'] = 119.9
dic['tres_C'] = 84
dic['bob_m3m3'] = 1.138
dic['muob_cP'] = 9.25

pvt_fields['MS0613'] = dic # MS0613	Романовское

dic = {}
dic['gamma_gas'] = 0.712
dic['gamma_oil'] = 0.86
dic['gamma_wat'] = 1.009
dic['rsb_m3m3'] = 15.6
dic['rp_m3m3'] = 53.2
dic['pb_atma'] = 119.9
dic['tres_C'] = 84
dic['bob_m3m3'] = 1.138
dic['muob_cP'] = 9.25

pvt_fields['MS0613'] = dic # MS0237	Муравленковское здесь просто копия от Романовского

''' Частота работы насоса ЭЦН по умолчанию '''
esp_freq_hz_default = 50