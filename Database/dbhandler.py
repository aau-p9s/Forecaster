from multiprocessing import Lock
import traceback
import psycopg2

class DbConnection:
    def __init__(self, dbname, username, password, hostname, port):
        self.connection = psycopg2.connect(database=dbname, user=username, password=password, host=hostname, port=port)
        self.lock = Lock()
        self.database = dbname
        self.user = username
        self.password = password
        self.host = hostname
        self.port = port
    def execute(self, query_string, params=None) -> None:
        print("Getting lock", flush=True)
        self.lock.acquire()
        print("Got lock", flush=True)
        if self.connection.closed:
            self.__init__(self.database, self.user, self.password, self.host, self.port)
        try:
            cursor = self.connection.cursor()
            if params == None:
                cursor.execute(query_string)
            else:
                cursor.execute(query_string, params)
            self.connection.commit()
        except Exception:
            print("There was an error in the database", flush=True)
            print(traceback.format_exc())
            cursor = self.connection.cursor()
            cursor.execute("ROLLBACK")
            self.connection.commit()
        self.lock.release()
        print("Released lock")

    def execute_get(self, query_string, params=None) -> list[tuple]:
        """Executes query given a querystring and optional parameters.
        Args:
            query_string: Sql query statement
            params: Query parameters
        """
        print("Getting lock", flush=True)
        self.lock.acquire()
        print("Got lock", flush=True)
        if self.connection.closed:
            self.__init__(self.database, self.user, self.password, self.host, self.port)
        try:
            cursor = self.connection.cursor()
            if params == None:
                cursor.execute(query_string)
            else:
                cursor.execute(query_string, params)
            data = cursor.fetchall()
            self.connection.commit()
            self.lock.release()
            print("Released lock", flush=True)
            return data
        except Exception:
            print("There was an error in the database", flush=True)
            print(traceback.format_exc())
            cursor = self.connection.cursor()
            cursor.execute("ROLLBACK")
            self.connection.commit()
            self.lock.release()
            print("Released lock", flush=True)
            return []

    
    def close(self):
        """Closes the database connection."""
        self.connection.close()
