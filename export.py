from ultralytics import YOLO
import wget
import os

site_url = ''
folder_path = './'  # 設定要刪除檔案的資料夾路徑

for filename in os.listdir(folder_path):
    if filename.endswith(".pt") or filename.endswith(".onnx"):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)


file_name = wget.download(site_url)
model = YOLO(file_name) 
path = model.export(format="onnx")