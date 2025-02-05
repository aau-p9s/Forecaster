from REST.api import start_api
from Database.dbhandler import DbConnection

if __name__ == '__main__':
    db = DbConnection("p10s", "postgres", "password", "localhost", 5432)
    # start_api()
    # users = db.get_all()
    # print(users)
    model_path = "Assets\\autotheta_model.pth"
    res = db.insert("AutoThetaTest", model_path)
    print(f"Inserted: {res}")