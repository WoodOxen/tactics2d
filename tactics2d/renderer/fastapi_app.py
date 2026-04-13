# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""fastapi app implementation."""

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
