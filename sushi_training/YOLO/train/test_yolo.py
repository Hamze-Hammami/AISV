from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

results = model(source='sim04.mp4', show=True)
