#----- Draws Bounding boxes on an image (created to see if yolo_format_converter.py does its job right) ----#

import cv2 # type: ignore

IMAGE_PATH = "./data/yolo/images/00899.jpg"     
LABEL_PATH = "./data/yolo/labels/00899.txt"    

# Load the image
img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]

# Load the label
with open(LABEL_PATH, "r") as f:
    for line in f:
        parts = line.strip().split()
        class_id, x_center, y_center, bbox_w, bbox_h = map(float, parts)
        # YOLO-Format -> coordinates
        x = int((x_center - bbox_w / 2) * w)
        y = int((y_center - bbox_h / 2) * h)
        x2 = int((x_center + bbox_w / 2) * w)
        y2 = int((y_center + bbox_h / 2) * h)
        # Draw rectangle
        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
        # Display the label
        cv2.putText(img, str(int(class_id)), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# Save or show the image
# cv2.imwrite("bild_mit_boxes.jpg", img)
cv2.imshow("YOLO Bounding Boxes", img)
cv2.waitKey(0)
# cv2.destroyAllWindows()
