
import oper_loader_and_encoder

svod_data = oper_loader_and_encoder.load_svod(
    'data_loader/20191202_dump__svod_opers_v.csv',
    'data_loader/20191202_dump__svod_opers_parms_v.csv',
    'data_loader/20191202_dump__repairs_v.csv',
    True,
    [],
    False,
    'data_loader/out_opers_code.csv',
    'data_loader/out_params_code.csv',
    'data_loader/out_params_values_code.csv',
    'data_loader/out_repairs_code.csv')

oper_loader_and_encoder.dump_loaded_svod_data(svod_data, 'data_loader/out_loaded_svod_data.csv')

svod_data = oper_loader_and_encoder.load_svod_data_dump('data_loader/out_loaded_svod_data.csv')

print('done')