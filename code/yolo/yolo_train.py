from ultralytics import YOLO # type: ignore

def main():
    # Load a YOLO model
    model = YOLO("yolo11s.pt")
    config_path = "D:/IU/s3/Projekt Computer Vision/Traffic Sign Detector/code/yolo/config.yaml"

    model.train(data=config_path, epochs=50, batch=16, imgsz=640)
    
if __name__ == "__main__":
    main()