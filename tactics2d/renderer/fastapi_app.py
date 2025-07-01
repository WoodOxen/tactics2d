# from fastapi import FastAPI, WebSocket

# app = FastAPI()


# @app.get("/health")
# def health():
#     return {"status": "running"}


# @app.post("/send_data")
# async def send_data(data):
#     msg = data


# @app.websocket("/ws")
# async def websocket_endpoint(ws):
#     await ws.accept()
#     while True:
#         data = await ws.receive_text()
#         await ws.send_text(f"Received: {data}")
