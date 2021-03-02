-- 2020-06-18(4th_repair)_dump_repairs_v.csv
select t.ID, t.WORK_TYPE_ID
from (select rownum as rn, t.*
      from (
               select r.ID,
                      r.WORK_TYPE_ID
               from REPAIRS r

                        inner join op_oper oo
                                   on r.work_type_id = oo.id

                        inner join op_groups og
                                   on oo.group_id = og.id
                                       and og.DICT_ID = 3
               where r.WELL_ID is not null
                 and r.STATE = 2
                 and r.TYPE_REPAIR in (0, 1, 2, 6)
                 and r.FACT_END_DATE > to_date('01.02.2020', 'DD.MM.YYYY')
               order by r.ID) t
     ) t
where rn - trunc(rn / 4) * 4 = 0;

-- 2020-06-18(4th_repair)_dump_opers_v.csv
select so.ID                                      as svod_opers_id,
       --so.FROMDATE,
       --so.TODATE,
       round((so.TODATE - so.FROMDATE) * 1440, 0) as duration_hours,
       so.NM_OPTEMPL_ID                           as op_templ_id,
       --so.PRODUCT_TIME,
       so.REPAIR_ID,
       so.FROMDATE
from REPAIRS r

         inner join op_oper oo
                    on r.work_type_id = oo.id

         inner join op_groups og
                    on oo.group_id = og.id
                        and og.DICT_ID = 3

         inner join SVOD_OPERS so
                    on r.ID = so.REPAIR_ID
where so.IS_HST = 0
  and so.NM_OPTEMPL_ID is not null
  and r.WELL_ID is not null
  and r.STATE = 2
  and r.TYPE_REPAIR in (0, 1, 2, 6)
--         and r.FACT_END_DATE > to_date(''01.02.2020'', ''DD.MM.YYYY'')
  and r.ID in (select t.ID
               from (select rownum as rn, t.*
                     from (
                              select r.ID,
                                     r.WORK_TYPE_ID
                              from REPAIRS r

                                       inner join op_oper oo
                                                  on r.work_type_id = oo.id

                                       inner join op_groups og
                                                  on oo.group_id = og.id
                                                      and og.DICT_ID = 3
                              where r.WELL_ID is not null
                                and r.STATE = 2
                                and r.TYPE_REPAIR in (0, 1, 2, 6)
                                and r.FACT_END_DATE > to_date('01.02.2020', 'DD.MM.YYYY')
                              order by r.ID) t
                    ) t
               where rn - trunc(rn / 4) * 4 = 0)
order by REPAIR_ID, FROMDATE;

-- 2020-06-18(4th_repair)_dump_opers_parms_v.csv
with opers_v as (
    -- opers
    select so.ID                                      as svod_opers_id,
           --so.FROMDATE,
           --so.TODATE,
           round((so.TODATE - so.FROMDATE) * 1440, 0) as duration_hours,
           so.NM_OPTEMPL_ID                           as op_templ_id,
           so.REPAIR_ID
    from REPAIRS r

             inner join op_oper oo
                        on r.work_type_id = oo.id

             inner join op_groups og
                        on oo.group_id = og.id
                            and og.DICT_ID = 3

             inner join SVOD_OPERS so
                        on r.ID = so.REPAIR_ID
    where so.IS_HST = 0
      and so.NM_OPTEMPL_ID is not null
      and r.WELL_ID is not null
      and r.STATE = 2
      and r.TYPE_REPAIR in (0, 1, 2, 6)
--         and r.FACT_END_DATE > to_date(''01.02.2020'', ''DD.MM.YYYY'')
      and r.ID in (select t.ID
               from (select rownum as rn, t.*
                     from (
                              select r.ID,
                                     r.WORK_TYPE_ID
                              from REPAIRS r

                                       inner join op_oper oo
                                                  on r.work_type_id = oo.id

                                       inner join op_groups og
                                                  on oo.group_id = og.id
                                                      and og.DICT_ID = 3
                              where r.WELL_ID is not null
                                and r.STATE = 2
                                and r.TYPE_REPAIR in (0, 1, 2, 6)
                                and r.FACT_END_DATE > to_date('01.02.2020', 'DD.MM.YYYY')
                              order by r.ID) t
                    ) t
               where rn - trunc(rn / 4) * 4 = 0)
    order by REPAIR_ID,
             FROMDATE
)
-- parms
select sp.SVOD_OPERS_ID,
       sp.NM_PARMS_ID as param_id,
       sp.PARAM_VALUE
from opers_v so

         inner join SVOD_OPERS_PARMS sp
                    on so.svod_opers_id = sp.SVOD_OPERS_ID

         inner join OP_PARAM p
                    on sp.NM_PARMS_ID = p.ID
                        and p.PARAM_TYPE in (
                            --0, -- text
                                             1, -- number
                                             2, -- boolean
                                             3);