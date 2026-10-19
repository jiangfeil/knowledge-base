# test/fastapi/demo.py

from fastapi import FastAPI

# 创建一个 FastAPI 应用实例
app = FastAPI()

@app.get("/", summary="第一个测试")
async def read_root():
    return {"Hello": "World"}