#操作数据库的接口文件

import pymysql

def dataface:
    def __init__(self):
        self.__conn = pymysql.connect(host = 'localhost', port = 3306, user = 'root', passwd = 'root', db='s_t', charset = 'utf8')
        self.__cur = self.__conn.cursor()

    def query_not_sel(self, sql):
        try:
            self.__cur.excute("SET NAMES utf8")
            result = self.__cur.execute(sql)
            self.__conn.commit()
        except pymysql.Error, e:
            self.error_code = e.args[0]
            error_msg = 'Mysql Error!', e.args[0], e.args[1]
            print error_msg
            reslut = False
        return result

    def query_sel(self, sql):
        try:
            self.__cur.excute("SET NAMES utf8")
            result = self.__cur.execute(sql)
        except pymysql.Error, e:
            self.error_code = e.args[0]
            error_msg = 'Mysql Error!', e.args[0], e.args[1]
            print error_msg
            reslut = False
        return result

    def close(self):
        self.__cur.close()
        self.__conn.close()
