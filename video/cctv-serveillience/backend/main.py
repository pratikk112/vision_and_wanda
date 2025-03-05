from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "CCTV Surveillance Backend is Running"}
