from multiprocessing import Lock
import traceback
import psycopg2

class DbConnection:
    def __init__(self, dbname, username, password, hostname, port):
        self.connection = psycopg2.connect(database=dbname, user=username, password=password, host=hostname, port=port)
        self.lock = Lock()
    def execute(self, query_string, params=None) -> None:
        self.lock.acquire()
        try:
            cursor = self.connection.cursor()
            if params == None:
                cursor.execute(query_string)
            else:
                cursor.execute(query_string, params)
            self.connection.commit()
        except Exception:
            print("There was an error in the database")
            print(traceback.format_exc())
            cursor = self.connection.cursor()
            cursor.execute("ROLLBACK")
            self.connection.commit()
        self.lock.release()

    def execute_get(self, query_string, params=None) -> list[tuple]:
        """Executes query given a querystring and optional parameters.
        Args:
            query_string: Sql query statement
            params: Query parameters
        """
        self.lock.acquire()
        try:
            cursor = self.connection.cursor()
            if params == None:
                cursor.execute(query_string)
            else:
                cursor.execute(query_string, params)
            data = cursor.fetchall()
            self.connection.commit()
            self.lock.release()
            return data
        except Exception:
            print("There was an error in the database")
            print(traceback.format_exc())
            cursor = self.connection.cursor()
            cursor.execute("ROLLBACK")
            self.connection.commit()
            self.lock.release()
            return []

    
    def close(self):
        """Closes the database connection."""
        self.connection.close()
