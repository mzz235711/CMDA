import json
import time
import psycopg2
import storage

PG_CONNECTION_STR = "dbname=dsb user=peizhi host=localhost port=5434"
def run_query(sql, bao_select=False, bao_reward=False):
    avg_time = 0
    for i in range(3):
        while True:
            try:
                conn = psycopg2.connect(PG_CONNECTION_STR)
                cur = conn.cursor()
                #cur.execute(f"SET enable_bao TO True")
                #cur.execute(f"SET enable_bao_selection TO False")
                #cur.execute(f"SET enable_bao_rewards TO False")
                #cur.execute("SET bao_num_arms TO 5")
                #cur.execute("SET statement_timeout TO 900000")
                cur.execute(sql)
                result = cur.fetchall()
                conn.close()
                avg_time += float(result[-1][0][16:-3])
                break
            except:
                print("sleep")
                time.sleep(1)
                continue
    avg_time /= 3
    return avg_time

dataset = "dsb"
query_path = "prediction"
infile = open("workloads/{}/{}.sql".format(dataset, query_path), 'r')
outfile = open("workloads/{}/{}_time.txt".format(dataset, query_path), 'w')
sqls = infile.readlines()
for i in range(len(sqls)):
    sql = sqls[i]
    sql = "explain analyze " + sql
    print(i, sql)
    result = run_query(sql)
    print(result)
    outfile.write("{}\n".format(result))
    outfile.flush()
    #all_experience = storage.experience()
    #print(all_experience[0])
