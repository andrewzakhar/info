create table esp_analysis_result
(
  well_id   	number		not null,
  param_id		number		not null,
  dt_from 		date		not null,
  dt_to			date		not null
);

alter table esp_analysis_result add (
  constraint esp_analysis_result_pk
  primary key (well_id, param_id, dt_from, dt_to));

comment on table esp_analysis_result is 'Таблица для хранения результатов анализа';
comment on column esp_analysis_result.well_id is 'Айди скважины';
comment on column esp_analysis_result.param_id is 'Айди аномалии из справочника eps_index';
comment on column esp_analysis_result.dt_from is 'Левая граница интервала аномалии';
comment on column esp_analysis_result.dt_to is 'Правая граница интервала аномалии';

grant select, insert, update, delete on esp_analysis_result to wellop;