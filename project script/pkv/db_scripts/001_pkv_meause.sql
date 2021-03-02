create table ESP_MEASURE
(
  WELL_ID   number                              not null,
  PARAM_ID  number                              not null,
  DT        date                                not null,
  VAL       number
);

alter table ESP_MEASURE add (
  constraint ESP_MEASURE_PK
  primary key
  (WELL_ID, PARAM_ID, DT));

grant select, insert, update, delete on quality_index to WELLOP;


comment on table esp_measure is 'Таблица для хранения значений телеметрии';
comment on column esp_measure.well_id is 'Айди скважины';
comment on column esp_measure.param_id is 'Айди параметра';
comment on column esp_measure.dt is 'Время';
comment on column esp_measure.val is 'Значение';