delete 
from esp_index
where id between 6 and 37;

insert into esp_index(id, name, shortname)
values(6, 'Поступление газа по току фазы А', '');
insert into esp_index(id, name, shortname)
values(7, 'Доля нестабильных режимов по току фазы А', '');
insert into esp_index(id, name, shortname)
values(8, 'Количество блоков из-за газа по току фазы А', '');
insert into esp_index(id, name, shortname)
values(9, 'Медианное количество точек для периода работы по току фазы А', '');
insert into esp_index(id, name, shortname)
values(10, 'Медианное количество точек для анализа по току фазы А', '');
insert into esp_index(id, name, shortname)
values(11, 'Рекомендуемое (медианное) время уменьшения работы по току фазы А', '');
insert into esp_index(id, name, shortname)
values(12, 'Медианное итоговое падение параметра по току фазы А, (обезразм), д.ед.', '');
insert into esp_index(id, name, shortname)
values(13, 'Медианное итоговое падение параметра по току фазы А, (относит), %', '');
insert into esp_index(id, name, shortname)
values(14, 'Медианное время работы по току фазы А, мин', '');
insert into esp_index(id, name, shortname)
values(15, 'Медианное время накопления по току фазы А, мин', '');
insert into esp_index(id, name, shortname)
values(16, 'Рабочая доля времени за весь период по току фазы А', '');
insert into esp_index(id, name, shortname)
values(17, 'Количество больших остановок по току фазы А', '');
insert into esp_index(id, name, shortname)
values(18, 'Поступление газа по загрузке двигателя, раз', '');
insert into esp_index(id, name, shortname)
values(19, 'Доля нестабильных режимов по загрузке двигателя, %', '');
insert into esp_index(id, name, shortname)
values(20, 'Количество блоков из-за газа по загрузке двигателя, раз', '');
insert into esp_index(id, name, shortname)
values(21, 'Медианное количество точек для периода работы по загрузке двигателя, штук', '');
insert into esp_index(id, name, shortname)
values(22, 'Медианное количество точек для анализа по загрузке двигателя', '');
insert into esp_index(id, name, shortname)
values(23, 'Рекомендуемое (медианное) время уменьшения работы по загрузке двигателя, мин', '');
insert into esp_index(id, name, shortname)
values(24, 'Медианное итоговое падение параметра по загрузке двигателя, (обезразм) д.ед.', '');
insert into esp_index(id, name, shortname)
values(25, 'Медианное итоговое падение параметра по загрузке двигателя, (относит) %.', '');
insert into esp_index(id, name, shortname)
values(26, 'Медианное время работы по загрузке двигателя, мин', '');
insert into esp_index(id, name, shortname)
values(27, 'Медианное время накопления по загрузке двигателя, мин', '');
insert into esp_index(id, name, shortname)
values(28, 'Рабочая доля времени за весь период по загрузке двигателя, %', '');
insert into esp_index(id, name, shortname)
values(29, 'Количество больших остановок по загрузке двигателя, %', '');
insert into esp_index(id, name, shortname)
values(30, 'Количество запусков с Загрузкой меньше 45%', '');
insert into esp_index(id, name, shortname)
values(31, 'Количество неудачных запусков с Загрузкой меньше 45%', '');
insert into esp_index(id, name, shortname)
values(32, 'Количество клинов (первым методом)), штук', '');
insert into esp_index(id, name, shortname)
values(33, 'Режимное значение параметра по току фазы А', '');
insert into esp_index(id, name, shortname)
values(34, 'Количество клинов (вторым методом), штук', '');
insert into esp_index(id, name, shortname)
values(35, 'Количество подклинок, штук', '');
insert into esp_index(id, name, shortname)
values(36, 'Число случаев турбинного вращения', '');
insert into esp_index(id, name, shortname)
values(37, 'Затрачено времени на анализ, сек', '');
insert into esp_index(id, name, shortname)
values(38, 'Режим работы', '');
commit;