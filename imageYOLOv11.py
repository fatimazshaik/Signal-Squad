# TEST 1 - checking if model can work
# from ultralytics import YOLO

# model = YOLO("yolo11n.pt")  # initialize model
# results = model("imgs/beach.jpg")  # perform inference
# results[0].show()  # display results for the first image

#---ACTUAL CODE---#

import cv2
from ultralytics import YOLO

# Notes:
# to change path of the input image use image var
# to change path of the ouput image  use output_image var

model = YOLO("yolo11x.pt")

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

# read the image
image = cv2.imread("imgs/beach.jpg") #change to image input path
result_img, _ = predict_and_detect(model, image, classes=[], conf=0.5)
output_image = "works.jpg" #change image output path

cv2.imwrite(output_image, result_img) 