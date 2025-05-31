import os
import cloudpickle
import psycopg2
from uuid import uuid4
from datetime import datetime
import argparse
import sys

parser = argparse.ArgumentParser(sys.argv[1])
parser.add_argument("--dbname", type=str, default="autoscaler")
parser.add_argument("--dbuser", type=str, default="root")
parser.add_argument("--dbpassword", type=str, default="password")
parser.add_argument("--dbhost", type=str, default="0.0.0.0")
parser.add_argument("--dbport", type=int, default=5432)
args = vars(parser.parse_args(sys.argv[1:]))

BASE_DIR = os.path.expanduser("./Assets/models")

DB_CONFIG = {
    "dbname": args["dbname"],
    "user": args["dbuser"],
    "password": args["dbpassword"],
    "host": args["dbhost"],
    "port": args["dbport"],
}

INSERT_SQL = '''
    INSERT INTO models (id, name, bin, trainedat, serviceid)
    VALUES (%s, %s, %s, %s, %s)
'''

def get_ids(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT id from services")
        rows = cursor.fetchall()
        return [row[0] for row in rows]



def insert_model(conn, name, binary, service_id):
    with conn.cursor() as cur:
        cur.execute(
            INSERT_SQL,
            [str(uuid4()), name, binary, datetime.utcnow(), service_id]
        )
        conn.commit()
        print(f"Inserted model: {name}")

def main():
    conn = psycopg2.connect(**DB_CONFIG)

    for model_name in os.listdir(BASE_DIR):
        model_dir = os.path.join(BASE_DIR, model_name)
        if not os.path.isdir(model_dir):
            continue

        # Search for .pth file
        pth_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
        if not pth_files:
            print(f"No .pth file in {model_name}")
            continue

        pth_path = os.path.join(model_dir, pth_files[0])

        try:
            with open(pth_path, "rb") as f:
                model = cloudpickle.load(f)
                print(f"Loaded model: {model_name}")
                binary = cloudpickle.dumps(model)
                for s in get_ids(conn):
                    insert_model(conn, model_name, binary, s)
        except Exception as e:
            print(f"Failed to process {model_name}: {e}")

    conn.close()

if __name__ == "__main__":
    main()
