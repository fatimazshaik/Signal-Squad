import os
import cv2
import sys

import numpy as np
import time

# Convert's bounding box coordinates ([x_min, y_min, width, height]) to the [x_min, y_min, x_bottom_right, y_bottom_right] format
def bbConvertToSortCoords(boundingbox):
    x_min = boundingbox[0]
    y_min = boundingbox[1]
    width = boundingbox[2]
    height = boundingbox[3]
    x_bottom_right = x_min + width
    y_bottom_right = y_min + height

    return [int(x_min), int(y_min), int(x_bottom_right), int(y_bottom_right)]

# Convert a list into a dictionary
def makeListToDict(list):
    dictionary = {}
    for x in range(len(list)):
        if list[x] in dictionary:
            dictionary[list[x]] = dictionary[list[x]] + 1  
        else:
            dictionary[list[x]] = 1
    return dictionary

# Convert a list of lists into a dictionary
def makeListOfListToDict(list):
    dictionary = {}
    combinedList = []
    for l in list:
        combinedList.extend(l)
    numElements = len(combinedList)
    return makeListToDict(combinedList), numElements

# Read classes from provided labels file --> Output a list of all classes
def getClasses(path_labels):    
    names = open(path_labels, "r")
    contents_names = names.read()
    classes = contents_names.split("\n")
    classes = classes[0: len(classes)-1]
    names.close()
    return classes

# Extract video name from path provided
def getVideoName(path_image):
    video_dir_list = path_image.split("/")
    video_title = video_dir_list[len(video_dir_list)-1]
    video_title_list = video_title.split(".")
    video_name = video_title_list[0]
    return video_name

# Draw bounding boxes for each identified object
def drawBoundingBoxes(output, probability_minimum, w, h):
    bounding_boxes = []
    confidences = []
    classes = []
    for result in output:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                classes.append(class_current)
    return bounding_boxes, confidences, classes

# Identify people and display people id per bounding box
def annotatingImages(results,bounding_boxes, colours, classes, image, labels, confidences, frame_height, frame_width, mot_tracker, max_people, video_name):
    # Get Detections:
    # detections = []
    # for i in results.flatten():
    #     if(labels[classes[i]] == "person"):
    #         detection =bbConvertToSortCoords(bounding_boxes[i])
    #         detection.append(confidences[i])
    #         detections.append(detection) # list of all detctions in proper format
    # detections_np = np.array(detections)
    # text_box_person_count = "hello"

    # # OG MOTS Tracker
    # track_bbs_ids = mot_tracker.update(detections_np)
    # for track_obj in track_bbs_ids:

    #     x_min, y_min = int(track_obj[0]), int(track_obj[1])
    #     box_width, box_height = int(track_obj[2]-track_obj[0]), int(track_obj[3]-track_obj[1])
    #     colour_box = [int(j) for j in colours[0]]

    #     id = track_obj[4]
    #     max_people = int(max(max_people, id))
    #     # text_box_person_count = 'People :' + str(max_people)
    #     cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box, 5)
    #     cv2.putText(image, str(int(id)), (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour_box, 5)
    #New Code want to Use:
    list_of_classes = []
    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box = [int(j) for j in colours[classes[i]]]
            cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height),
                        colour_box, 5)
            text_box = labels[classes[i]] + ': {:.4f}'.format(confidences[i])
            list_of_classes.append(labels[classes[i]])
            cv2.putText(image, text_box, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour_box, 5)

    return image, max_people

# Feed the each frame of a video through the YOLO Object Detection Network
def runYOLO(path_to_cfg, path_to_weights, path_video, path_labels):

    #defining information variables
    inference_time = 0

    #get labels from file
    labels = getClasses(path_labels)

    # load yolo model
    network = cv2.dnn.readNetFromDarknet(path_to_cfg, path_to_weights)
    layers = network.getLayerNames()
    yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']

    #Make sort
    # mot_tracker = Sort() 

    # load video
    cap = cv2.VideoCapture(path_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("mot_vid/result.mp4", fourcc, fps, (frame_width, frame_height)) #change video name

    while cap.isOpened():
        ret, frame = cap.read()
 
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            print(ret)
            break

        # convert image to a blob
        input_blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        network.setInput(input_blob)
        output = network.forward(yolo_layers)

        # define variables for bounding boxes, confidence level, inference time, max id detected
        inference_time_frame =  0
        max_people = 0
        bounding_boxes = []
        confidences = []
        classes = []
        probability_minimum = 0.5
        threshold = 0.3
        h, w = frame.shape[:2]

        # draw and show bounding boxes
        bounding_boxes, confidences, classes = drawBoundingBoxes(output, probability_minimum,  w, h)

        # include class names by index
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
        coco_labels = 80
        np.random.seed(42)
        colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')
        mot_tracker = 0 #intilaize so dont get issue

        annotated_frame, max_people = annotatingImages(results,bounding_boxes, colours, classes, frame, labels, confidences, frame_height, frame_width, mot_tracker, max_people, video_name='video')
        inference_time_frame = time.time() - inference_time_frame
        inference_time += inference_time_frame
        out.write(annotated_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return inference_time, max_people

# Run's YOLO and people identification on a single video if provided
def isSingleVideo(argv, list_inference_time):
    inf_time, max_people = runYOLO(path_to_cfg = argv[1], path_to_weights = argv[2], 
                                              path_video= argv[3], path_labels = argv[4])
    list_inference_time.append(inf_time) 
    return max_people

# Helper function to read help.txt and display program functionality
def filePrint(filename, start, end):
    i = 0
    with open(filename, 'r') as f:
        for line in f:
            i = i + 1
            if i < start or i > end:
                continue
            else:
                print(line, end='')

# Function to display help parameters based on user input: run, input_flag, output_flag
def printHelp():
    filePrint('help.txt', 52, 69)
    user = input("Type for more info: ")
    if user == "run":
        filePrint('help.txt', 69, 76)
    elif user == "output_flag":
        filePrint('help.txt', 76, 87)        

# Parse for output flag and print relevant information about object detections
def outputFlag(args, list_inference_time, max_people):
    for x in args:
        if x == "-h":
            printHelp()
        elif x == "-inf": 
            print(f"Average Inference: {np.mean(list_inference_time):4f} seconds")
        elif x=="-people_all":
            print("Total Number of People Detected: ", int(max_people))
        print()


# main function to specify program flow
if __name__ == '__main__':
    if(len(sys.argv)< 4):
        print("Need more arguements! Use -h to get show functionality")   
        #Format is: path_to_cfg path_to_weights path_video path_labels -other_flags.
        
    else:
        #Variables Storing Key information:
        inference_time = []
        max_people = 0

        #Run Detection & Yolo on Inputed Video
        max_people = isSingleVideo(sys.argv, inference_time)

        # check output flag to determine what to print: -inf (= inference)
        outputFlag(sys.argv[4:], inference_time, max_people)