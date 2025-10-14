from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncpg
from typing import List, Dict, Any, Union, Optional

import os # 환경 변수를 읽기 위해 os 모듈 추가

app = FastAPI()

@app.get("/")
def default_get(mydata: Optional[str] = None,
                mydata2: Optional[str] = None):
    result={"success":True, "data":None, "msg":""}
    try:
        result["data"]=f"mydata : {mydata}, mydata2:{mydata2}"
        return result
    except Exception as e:
        result["success"]=False
        result["msg"]=f"server error. {e!r}"
        return result
    

class Item(BaseModel):
    name:str
    price:int
@app.post("/item")
def post_item(item:Item):
    result={"success":True, "data":None, "msg":""}
    try:
        result["data"]=item
        return result
    except Exception as e:
        result["success"]=False
        result["msg"]=f"server error. {e!r}"
        return result

if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)