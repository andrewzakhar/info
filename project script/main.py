from common import conf
from readers import reader_mf as mf
from readers import reader_kolon
from readers import reader_incl
from readers import reader_pvt
from readers import reader_tr
import datetime
import pandas as pd

config = conf.conf_bd(None, None)

well_list = [2860198200, 2860193500, 2860197600, 2860048400, 2860065000, 2860081800,
    6110114200, 6110130800, 6110206400, 6110246600, 6110258600, 6110273800,
    6130103400, 6130105000, 6130117800, 6130212800, 6130214800, 6130092100, 6130094600, 6130097500]

reader = mf.reader_mf(config, well_list)
df = reader.df_source_data

print(df)
print('metadata = ', reader.metadata)
esp = reader.get_esp(2860198200);
print(esp.pump_name, esp.esp_nom_rate_m3day, esp.pump_depth_m)

reader_kolon_bd = reader_kolon.reader_kolon_ora(config, well_list)
print(reader_kolon_bd.df_source_data)
#kol = reader_kolon_bd.get_ekolon_row(2270142600, 1000)
#print('kol =', kol)
# df[lambda x: x['count'] > 15]
depth = 800

dek = reader_kolon_bd.get_ekolon_row(2860198200, depth)

print('dek =', dek)

reader_incl = reader_incl.reader_incl_ora(config, well_list)

print('reader_incl ***********')

dl = reader_incl.get_depths_list(2860198200)
dff = reader_incl.depth_angle_df
if dff is not None:
    angle = reader_incl.get_angle_depth(dl[5])
print(angle)

print('reader_pvt ***********')

reader_pvt = reader_pvt.reader_pvt_ora(config, well_list)

df = reader_pvt.df_source_data
print(df)

mpvt = reader_pvt.get_pvt(2860198200)

print(mpvt)

print('reader_tr ***********')

tr = reader_tr.reader_tr_ora(config, 2860198200, datetime.datetime(2020, 4, 23))

df = tr.df_source_data
print('Газовый фактор =', df['GAS_FACTOR'][2860198200])

#
# print('************')
# col = []
# ind = []
# dic = {}
# for r in c:
#     col.append(r[0])
#     ind.append(r[1])
# df = pd.DataFrame({m[0][0]: col}, index = ind)
# print(df['REGION_NAME'])
# dic.update({m[0][0]: col})
# print('dic =', dic)
#
# dic = {}
# col = []
# col_index = 0
# ind = []
# index_num_col = 1
# for col_name in m:
#     col_data = []
#     for col in c:
#         col_data.append(col[col_index])
#         if col_index == index_num_col:
#             ind.append(col[index_num_col])
#     dic.update({col_name[0]: col_data})
#     col_index += 1
#
# df = pd.DataFrame(dic, index = ind)
# print('************ 2')
#
# print(df)