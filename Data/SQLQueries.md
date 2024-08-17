# SQL Queries
### Receiving -> PTC
```
select ptt.whse,cntr_nbr, lh.locn_brcd as to_locn, lh2.locn_brcd as from_locn, ptt.nbr_of_cases,
ptt.nbr_units,ptt.menu_optn_name, to_char(ptt.create_date_Time,'yyyy-mm-dd hh24:mi') as create_Date, ptt.user_id 
from prod_trkg_tran ptt
inner join item_cbo ic on ic.item_id = ptt.item_id
inner join locn_hdr lh on lh.locn_id= ptt.to_locn and lh.work_grp = 'PTC'
left outer join locn_hdr lh2 on lh2.locn_id= ptt.from_locn 
where trunc(ptt.create_date_time) > sysdate - 120
order by ptt.create_date_time
```

### Receiving -> IBNP
```
"select th.whse,td.task_id, td.cntr_nbr, lh2.locn_brcd,lh.locn_brcd, td.stat_code, th.create_Date_time, th.mod_date_time,
case when substr(lh2.locn_brcd,4,1) = 'R' THEN 'Drop in IBNP' ELSE 'No Drop' end as test, th.useR_id
from task_hdr th 
inner join task_dtl td on td.task_id = th.task_id
inner join locn_hdr lh on lh.locn_id = td.dest_locn_id
inner join locn_hdr lh2 on lh2.locn_id = td.pull_locn_id
where th.invn_need_type = '11'
and td.stat_code = '90'
and th.whse = '800'
and th.create_Date_Time > sysdate - 120"
```

### PTC -> Shipping
```
"select cntr_nbr,ptt.module_name,ptt.menu_optn_name, 
lh.locn_brcd as from_locn, ptt.nbr_units,
lh2.locn_brcd as to_locn,ptt.user_id,
to_char(begin_date,'yyyy-mm-dd hh24:mi:ss') as begin_date
, to_char(end_date,'yyyy-mm-dd hh24:mi:ss') as end_date,
ptt.create_Date_time as record_created
from prod_trkg_tran ptt
inner join locn_hdr lh on lh.locn_id = ptt.from_locn and lh.locn_class = 'J'
inner join locn_hdr lh2 on lh2.locn_id = ptt.to_locn 
where menu_optn_name = 'BJAnchor oLPN'
and ptt.create_Date_time > sysdate -120"
```

### Xdock (Dock -> Dock)
```
WITH anchor_task AS (
    SELECT
        th.whse,
        td.task_id,
        td.cntr_nbr,
        lh.locn_brcd,
        td.stat_code,
        td.carton_nbr,
        th.create_date_time,
        th.mod_date_time,
        td.user_id
    FROM
             task_hdr th
        INNER JOIN task_dtl td ON td.task_id = th.task_id
        INNER JOIN locn_hdr lh ON lh.locn_id = td.dest_locn_id
    WHERE
            th.invn_need_type = '70'
        AND td.stat_code = '90'
        AND th.whse = '800'
        AND th.create_date_time > sysdate - 35
), cross_dock_task AS (
    SELECT
        aid.whse,
        aid.cntr_nbr,
        lh.locn_brcd,
        aid.stat_code,
        aid.carton_nbr,
        aid.create_date_time,
        aid.mod_date_time,
        aid.user_id
    FROM
        alloc_invn_dtl aid
        LEFT OUTER JOIN locn_hdr       lh ON lh.locn_id = aid.pull_locn_id
    WHERE
            aid.invn_need_type = '2'
        AND aid.stat_code = '90'
        AND aid.whse = '800'
    UNION
    SELECT
        th.whse,
        td.cntr_nbr,
        lh.locn_brcd,
        td.stat_code,
        td.carton_nbr,
        th.create_date_time,
        th.mod_date_time,
        td.user_id
    FROM
             task_hdr th
        INNER JOIN task_dtl td ON td.task_id = th.task_id
        INNER JOIN locn_hdr lh ON lh.locn_id = td.pull_locn_id
    WHERE
            th.invn_need_type = '2'
        AND td.stat_code = '90'
        AND th.whse = '800'
), prod_track AS (
    SELECT
        cntr_nbr,
        lh.locn_brcd,
        ptt.create_date_time,
        ptt.user_id
    FROM
             prod_trkg_tran ptt
        INNER JOIN locn_hdr lh ON lh.locn_id = ptt.from_locn
    WHERE
        menu_optn_name = 'BJLPN Disposition'
)
SELECT
    cdt.whse,
    cdt.cntr_nbr as ilpn,
    cdt.locn_brcd as locn_maybe,
    cdt.carton_nbr as olpn,
    cdt.create_date_time as cross_dock_create_date,
    cdt.mod_date_time as cross_dock_last_touched,
    pt.locn_brcd as original_locn,
    pt.create_date_time as Dispo_create_Date,
    pt.user_id as dispo_user,
    a.stat_code,
    a.mod_date_time as anchor_last_touched,
    a.locn_brcd as anchor_locn,
    a.user_id as anchor_user
    --,
    --cdt.*,
    --pt.*,
    --a.*
FROM
         cross_dock_task cdt
    INNER JOIN prod_track  pt ON pt.cntr_nbr = cdt.cntr_nbr
    INNER JOIN anchor_task a ON a.cntr_nbr = cdt.cntr_nbr
WHERE
    cdt.create_date_time > sysdate - 30
```