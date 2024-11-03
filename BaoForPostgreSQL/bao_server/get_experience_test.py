import psycopg2
import sys
import time
import storage
import json
import os

PG_CONNECTION_STR = "dbname=imdb user=peizhi host=localhost port=5434"
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
            cur.execute("SET statement_timeout TO 600000")
            cur.execute(sql)
            cur.fetchall()
            conn.close()
            break
        except:
            time.sleep(1)
            continue
    stop = time.time()
    return stop - start

dataset = "job-light-ranges"
query_path = "job-light_70"
infile = open("workloads/{}/{}.sql".format(dataset, query_path), 'r')
folder_path = "workloads/{}/{}_plans".format(dataset, query_path)
if os.path.exists(folder_path) is False:
    os.makedirs(folder_path)
queries = infile.readlines()
print("Read", len(queries), "queries.")

print("Executing queries using PG optimizer for initial training")

for q in queries:
    pg_time = run_query(q, bao_reward=True)
    print("x", "x", time.time(), pg_time, "PG", flush=True)
    break

all_experience = storage.experience()
print(all_experience)
#print(len(all_experience))
#for i, exp in enumerate(all_experience):
#    print(i)
#    outfile = "workloads/{}/{}_plans/sql{}_plan.json".format(dataset, query_path, i)
#    with open(outfile, 'w') as f:
#        json.dump(exp, f)
