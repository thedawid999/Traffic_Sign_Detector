from ultralytics import YOLO # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import torch # type: ignore
import time
import os


# prints a list of available GPUs
print("GPUs: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)


VIDEO = "C:/Users/dawid/Downloads/drive.mp4" # 0 to use camera, a path to use video
PIC = "./data/images/cnn/val/42/00004_00023.jpg"
YOLO_ONLY_MODEL = "./runs/detect/train6_yolo_only/weights/best.pt"
YOLO_MODEL = "./runs/detect/train6_yolo_cnn/weights/best.pt"
CNN_MODEL = "./code/cnn/cnn_classifier.h5"
INPUT_SIZE_CNN = 64

# YOLO_ONLY: Detect and classify traffic signs in a picture
def yolo_picture():
    yolo_model = YOLO(YOLO_ONLY_MODEL)

    img = cv2.imread(PIC)
    img_copy = img.copy()

    # get the prediction
    results = yolo_model.predict(source=img, verbose=False) 

    for result in results:
        for box in result.boxes:
            conf = box.conf[0]
            # out of all bounding boxes use only those with conf > 0.5
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                draw_bounding_box(img_copy, [x1,y1,x2,y2], cls, conf)

    cv2.imshow("YOLO Prediction", img_copy)
    # save the output
    cv2.imwrite("./outputs/detected_yolo_only.jpg", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# YOLO_ONLY: Detect and classify traffic signs live
def yolo_live():
    model = YOLO(YOLO_ONLY_MODEL)
    cap = cv2.VideoCapture(VIDEO)  

    # get video as a variable
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        './outputs/output_detected_yoloonly.mp4',
        fourcc,
        int(cap.get(cv2.CAP_PROP_FPS)),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    latencies = [] # for final stats
    fps_values = [] # for final stats
    frame_count = 0
    last_fps_time = time.time()

    print("Program starts. Press 'q' to quit")

    while True:
        # count time and frames
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        # copy each frame to write on it
        img_copy = frame.copy()

        # get the preditction
        results = model(frame)

        for result in results:
            for box in result.boxes:
                conf = box.conf[0]
                # out of all bounding boxes use only those with conf > 0.5
                if conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    draw_bounding_box(img_copy, [x1,y1,x2,y2], cls, conf)

        # calculate latency and save it in the list
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
        frame_count += 1

        # after one second, add the amount of displayed frames to the list
        if time.time() - last_fps_time >= 1.0:
                fps_values.append(frame_count)
                frame_count = 0
                last_fps_time = time.time()

        # display latency and fps live
        cv2.putText(img_copy, f"Latency: {latency:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if fps_values:
            cv2.putText(img_copy, f"FPS: {fps_values[-1]}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        cv2.imshow("Live YOLO Detection", img_copy)
        # save the video
        out.write(img_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video saved as output_detected.mp4")
    plot_latency_and_fps(latencies[1:], fps_values)

# YOLO+CNN: Detect and classify traffic signs in a picture
def picture():
    yolo = YOLO(YOLO_MODEL)
    cnn = load_model(CNN_MODEL)

    img = cv2.imread(PIC)
    img_copy = img.copy()

    # get the prediction
    results = yolo.predict(img, verbose=False)[0]

    # iterate through all predicted bounding boxes
    for box, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()):
        if conf < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box)

        # image crop for CNN
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # prapre the format for CNN
        crop_input = prepare_for_keras(crop, (INPUT_SIZE_CNN, INPUT_SIZE_CNN))
        crop_input = np.expand_dims(crop_input, axis=0)

        # CNN-Classification
        pred = cnn.predict(crop_input, verbose=0)
        pred_class = np.argmax(pred)

        # display bounding boxes + label
        draw_bounding_box(img_copy, [x1, y1, x2, y2], pred_class, conf)

    # show results
    cv2.imshow("YOLO + CNN Detection", img_copy)
    cv2.imwrite("./outputs/detected_yolo_cnn.jpg", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# YOLO+CNN: Detect and classify traffic signs live (draws latency and fps diagrams at the end)
def live():
    yolo_model = YOLO(YOLO_MODEL)
    cnn_model = load_model(CNN_MODEL)
    cap = cv2.VideoCapture(VIDEO)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./outputs/output_detected.mp4', fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    latencies = [] # for final stats
    fps_values = [] # for final stats
    frame_count = 0
    last_fps_time = time.time()

    print("Program starts. Press 'q' to quit")

    while True:
        # read camera frames
        start_time = time.time()  # Start Timer
        success, frame = cap.read()
        if not success:
            break

        # copy the frame to write on it
        img_copy = frame.copy()
        results = yolo_model(frame, device="cuda", verbose=False) # verbose=False keeps the console clean

        # get all predicted bounding boxes and its confidences
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        crops_input = []
        boxes_filtered = []

        for box, conf in zip(boxes, confidences):
            # classify every box with keras, that confidence is higher than 0.5
            if conf < 0.5:
                continue

            # crop
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            crop_input = prepare_for_keras(crop, (INPUT_SIZE_CNN, INPUT_SIZE_CNN))
            crops_input.append(crop_input)
            boxes_filtered.append((x1, y1, x2, y2, conf))

        # convert
        if len(crops_input) > 0:
            crops_input = np.array(crops_input) # shape: (N, 64, 64, 3)
            preds = cnn_model.predict(crops_input, batch_size=len(crops_input))
            # draw all bounding boxes
            for (x1, y1, x2, y2, conf), pred in zip(boxes_filtered, preds):
                pred_class = np.argmax(pred)
                draw_bounding_box(img_copy, [x1,y1,x2,y2], pred_class, conf)

        # calculate latency and save it in the list
        latency = (time.time() - start_time) * 1000  # latency in ms
        latencies.append(latency)

        frame_count += 1

        # after one second, add the amount of displayed frames to the list
        if time.time() - last_fps_time >= 1.0:
            fps_values.append(frame_count)
            frame_count = 0
            last_fps_time = time.time()

        # show latency and fps on screen
        cv2.putText(img_copy, f"Latency: {latency:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if fps_values:
            cv2.putText(img_copy, f"FPS: {fps_values[-1]}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Live-Detection", img_copy)
        out.write(img_copy)
        # end the loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video saved as output_detected.mp4")
    plot_latency_and_fps(latencies[1:], fps_values)

# create a graph for live and yolo_live showing latency and fps
def plot_latency_and_fps(latencies, fps_values):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(latencies, label='Latency (ms)')
    plt.xlabel('Frames')
    plt.ylabel('Latency (ms)')
    plt.title('Latency per Frame')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fps_values, label='FPS')
    plt.xlabel('Seconds')
    plt.ylabel('FPS')
    plt.title('Frames per seconds')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Converts image size, normalizes pixel values and adds one additional dimension for batch (1, 64, 64, 3)
def prepare_for_keras(cropped_img, target_size):
    # set imgsz and normalize it
    img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(img_rgb, target_size)
    crop_normalized = crop_resized / 255.0
    # add batch dimension
    return crop_normalized.astype("float32")

# draws bounding boxes with given coords
def draw_bounding_box(img, box, cls, conf, color=(0,0,255)):
    x1, y1, x2, y2 = box
    text = f"{cls} {conf:.2f}"

    # calculate text size
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    # set the text position to "inside the bounding box"
    text_x = x1 + 2
    text_y = y1 + text_height + 2 

    # display bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # display background for label
    cv2.rectangle(
        img,
        (text_x, text_y - text_height - baseline),
        (text_x + text_width, text_y + baseline),
        color,
        thickness=-1, 
    )

    # display label
    cv2.putText(
        img,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

if __name__ == "__main__":
    yolo_picture()
    picture()