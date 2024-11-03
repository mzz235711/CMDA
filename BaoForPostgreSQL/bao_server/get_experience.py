import psycopg2
import sys
import time
import storage
import json
import os

PG_CONNECTION_STR = "dbname=dsb user=peizhi host=localhost port=5434"
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def run_query(sql, bao_select=False, bao_reward=False):
    start = time.time()
    while True:
        try:
            print(sql)
            conn = psycopg2.connect(PG_CONNECTION_STR)
            cur = conn.cursor()
            cur.execute(f"SET enable_bao TO True")
            cur.execute(f"SET enable_bao_selection TO False")
            cur.execute(f"SET enable_bao_rewards TO True")
            cur.execute("SET bao_num_arms TO 5")
            cur.execute("SET statement_timeout TO 900000")
            cur.execute(sql)
            cur.fetchall()
            conn.close()
            break
        except:
            time.sleep(1)
            #continue
            return -1
    stop = time.time()
    return stop - start

dataset = "dsb"
query_path = "prediction"
infile = open("workloads/{}/{}.sql".format(dataset, query_path), 'r')
folder_path = "workloads/{}/{}_plans".format(dataset, query_path)
if os.path.exists(folder_path) is False:
    os.makedirs(folder_path)
queries = infile.readlines()
print("Read", len(queries), "queries.")

print("Executing queries using PG optimizer for initial training")
all_time = []

for q in queries:
    pg_time = run_query(q, bao_reward=True)
    all_time.append(pg_time)
    print("x", "x", time.time(), pg_time, "PG", flush=True)

all_experience = storage.experience()
print(len(all_experience))
idx = 0
timefile = open("workloads/{}/{}_time.txt".format(dataset, query_path), 'w')
for i, exp in enumerate(all_experience):
    if all_time[i] > 0:
        print(idx)
        outfile = "workloads/{}/{}_plans/sql{}_plan.json".format(dataset, query_path, idx)
        with open(outfile, 'w') as f:
            json.dump(exp, f)
        timefile.write(str(all_time[i] * 1000) + "\n")
        idx += 1
