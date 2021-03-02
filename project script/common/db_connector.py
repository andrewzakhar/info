#import config
#from common import utils

#utils.set_oracle_path()

import cx_Oracle

def initSession(connection, requestedTag):
    module = 'itsk.wellop.vr'
    cursor = connection.cursor()
    cursor.execute("begin dbms_application_info.set_module(:1,:2); end;", [module, module])
    cursor.close()

# def get_connection_single():
#     global _ora_connection
#     if not _ora_connection:
#         _ora_connection = cx_Oracle.connect(config.oracle['user'], config.oracle['password'], config.oracle['connection'], sessionCallback=initSession)
#     return _ora_connection

# def get_connection():
#     global _ora_connection
#     if not _ora_connection:
#         _ora_connection = cx_Oracle.SessionPool(config.oracle['user'], config.oracle['password'], config.oracle['connection'],
#                         min=2, max=5, increment=1, threaded=True, getmode=cx_Oracle.SPOOL_ATTRVAL_TIMEDWAIT, waitTimeout=3000, sessionCallback=initSession)
#         # SPOOL_ATTRVAL_TIMEDWAIT + waitTimeout = ждать (в миллисекундах), пока сеанс станет доступным в пуле, прежде чем вернуться с ошибкой
#     return _ora_connection.acquire()

def get_conn_single(user, passw, connection):
    _ora_connection = cx_Oracle.connect(user, passw, connection, encoding='UTF-8')
    return _ora_connection

def ora_execute_select(conn, sql: str, params: dict = {}):
    if conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        meta_data = cur.description
        cur.close()
        return rows, meta_data

def ora_execute_statement(conn, sql: str, params: dict = {}):
    if conn:
        cur = conn.cursor()
        try:
            cur.execute(sql, params)
            cur.close()
        finally:
            conn.commit()
        return True
