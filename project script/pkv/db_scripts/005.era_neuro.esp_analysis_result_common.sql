create table esp_analysis_result_common
(
  well_id   	number		    not null,
  dt_from 		date		    not null,
  dt_to			date		    not null,
  param_id	    number	        not null,
  value			float       	not null
);

alter table esp_analysis_result_common add (
  constraint esp_analysis_result_common_pk
  primary key (well_id, dt_from, dt_to));

comment on table esp_analysis_result_common is 'Таблица для хранения сводной информации по итогам анализа';
comment on column esp_analysis_result_common.well_id is 'Айди скважины';
comment on column esp_analysis_result_common.dt_from is 'Дата первого замера';
comment on column esp_analysis_result_common.dt_to is 'Дата последнего замера';
comment on column esp_analysis_result_common.param_id is 'Показатель';
comment on column esp_analysis_result_common.value is 'Значение';

grant select, insert, update, delete on esp_analysis_result_common to wellop;