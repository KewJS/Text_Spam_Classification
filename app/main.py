import pathlib
from fastapi import FastAPI
import uvicorn

app = FastAPI()

BASE_DIR = pathlib.Path(__file__).resolve().parent

@app.get("/")
def read_index():
    return {"hello": "world", "BASE_DIR": str(BASE_DIR)}