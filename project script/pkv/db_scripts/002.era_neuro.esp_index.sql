create table esp_index
(
  id   			number				not null,
  name  		varchar2(200)		not null,
  shortname 	varchar2(100)
);

alter table esp_index add (
  constraint esp_index_pk
  primary key (id));

comment on table esp_index is 'Таблица-справочник для хранения наименований аномалий';
comment on column esp_index.id is 'Идентификатор';
comment on column esp_index.name is 'Полное наименование';
comment on column esp_index.shortname is 'Краткое наименование';

grant select, insert, update, delete on esp_index to wellop;