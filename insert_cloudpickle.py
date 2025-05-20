import os
import cloudpickle
import psycopg2
from uuid import uuid4
from datetime import datetime


BASE_DIR = os.path.expanduser("./Assets/models")

DB_CONFIG = {
    "dbname": "autoscaler",
    "user": "root",
    "password": "password",
    "host": "localhost",
    "port": 5432,
}

INSERT_SQL = '''
    INSERT INTO models (id, name, bin, trainedat, serviceid)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING id
'''

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
                for s in ["1a2b3c4d-1111-2222-3333-444455556666", "2b3c4d5e-1111-2222-3333-444455556666", "3c4d5e6f-1111-2222-3333-444455556666"]:
                    insert_model(conn, model_name, binary, s)
        except Exception as e:
            print(f"Failed to process {model_name}: {e}")

    conn.close()

if __name__ == "__main__":
    main()
