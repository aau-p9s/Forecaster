import psycopg2

class DbConnection:
    def __init__(self, dbname, username, password, hostname, port):
        self.connection = psycopg2.connect(database=dbname, user=username, password=password, host=hostname, port=port)
        self.cursor = self.connection.cursor()
    
    def execute_query(self, query_string, params=None):
        """Executes query given a querystring and optional parameters.
        Args:
            query_string: Sql query statement
            params: Query parameters
        """
        if params == None:
            self.cursor.execute(query_string)
        else:
            self.cursor.execute(query_string, params)
        data = self.cursor.fetchall()
        self.connection.commit()
        return data
    
    def close(self):
        """Closes the database connection."""
        self.connection.close()