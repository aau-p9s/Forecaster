import psycopg2

class DbConnection:
    def __init__(self, dbname, username, password, hostname, port):
        self.connection = psycopg2.connect(database=dbname, user=username, password=password, host=hostname, port=port)
    
    def execute_query(self, query_string, params=None):
        """Executes query given a querystring and optional parameters.
        Args:
            query_string: Sql query statement
            params: Query parameters
        """
        try:
            cursor = self.connection.cursor()
            if params == None:
                cursor.execute(query_string)
            else:
                cursor.execute(query_string, params)
            data = cursor.fetchall()
            self.connection.commit()
            return data
        except:
            print("There was an error in the database")
            cursor = self.connection.cursor()
            cursor.execute("ROLLBACK")
            self.connection.commit()
            return None

    
    def close(self):
        """Closes the database connection."""
        self.connection.close()
