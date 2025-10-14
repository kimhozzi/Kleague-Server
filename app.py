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

"""
t_board 용 데이터 받기
class Board 라는거 만들고,
user_id : int
title : str
content : str
받는 post router 만들기
함수 이름과 end point 는 post_board
"""

class Board(BaseModel):
    user_id:int
    title:str
    content:str

@app.post("/post_board")
def post_board(board:Board):
    result={"success":True, "data":None, "msg":""}
    try:
        result["data"]=board
        return result
    except Exception as e:
        result["success"]=False
        result["msg"]=f"server error. {e!r}"
        return result

if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)