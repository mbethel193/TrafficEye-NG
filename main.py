from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
import cv2

app = FastAPI(title="TrafficEye NG API", version="1.0.0")

# Initialize fine-tuned YOLOv8m model for Nigerian road classes
model = YOLO("weights/yolov8m_trafficeye_finetuned.pt")

@app.get("/")
async def root():
    return {"message": "TrafficEye NG Backend is running successfully."}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", model: "YOLOv8m + ByteTrack"}
