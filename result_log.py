import mysql.connector


def log_result(log_data):
    db_user = 'root'
    db_database = 'sharebox'
    language = 'EN'
    cnx = mysql.connector.connect(user=db_user, database=db_database)
    cursor = cnx.cursor(dictionary=True)

    columns = ','.join("`" + str(x).replace('/', '_') + "`" for x in log_data.keys())
    values = ','.join("'" + str(x).replace('/', '_') + "'" for x in log_data.values())
    sql = "INSERT INTO %s (test_time, %s) VALUES (NOW(), %s );" % ('result_log', columns, values)

    try:
        cursor.execute(sql)
        cnx.commit()
    except mysql.connector.Error as e:
        print("x Failed inserting data: {}\n".format(e))
