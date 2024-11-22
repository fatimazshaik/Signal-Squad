# yolo_img_detector.py
# (Phase 1) Use YOLO to detect object from image and display confidence and class label

'''To run: python yolo_img_detector.py yolo_files/yolov3.cfg yolo_files/yolov3.weights <-d/ -m/ none> <path_to_imgs> yolo_files/coco.names <-h/ -inf/ -tot_brkdwn/ -classes_all/ -tot_brkdwn_img>'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

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

# Count the number of valid .jpg images 
def getNumImages(arguments):
    num_imgs = 0
    for x in arguments[3:]:
        if x[-4:] == ".jpg":
            num_imgs +=1
    return num_imgs

# Read classes from provided labels file --> Output a list of all classes
def getClasses(path_labels):    
    names = open(path_labels, "r")
    contents_names = names.read()
    classes = contents_names.split("\n")
    classes = classes[0: len(classes)-1]
    names.close()
    return classes

# Extract image name from path provided
def getImageName(path_image):
    image_dir_list = path_image.split("/")
    image_title = image_dir_list[len(image_dir_list)-1]
    image_title_list = image_title.split(".")
    image_name = image_title_list[0]
    return image_name

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

# Identify class and display confidence level per bounding box
def annotatingImages(results,bounding_boxes, colours, classes, image, labels, confidences, image_name):
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

    cv2.imwrite('out_imgs/' + image_name + '_out.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return list_of_classes

# Feed the image through the YOLO Object Detection Network
def runYOLO(path_to_cfg, path_to_weights, path_image, path_labels):

    #defining information variables
    inference_time = time.time()

    #get labels from file
    labels = getClasses(path_labels)

    #get image name
    image_name = getImageName(path_image)

    # load yolo model
    network = cv2.dnn.readNetFromDarknet(path_to_cfg, path_to_weights)
    layers = network.getLayerNames()
    yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']

    # load an image
    image = cv2.imread(path_image)

    # convert image to a blob
    input_blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # pass the image through the network
    network.setInput(input_blob)
    output = network.forward(yolo_layers)

    # define variables for bounding boxes and confidence level
    bounding_boxes = []
    confidences = []
    classes = []
    probability_minimum = 0.5
    threshold = 0.3
    h, w = image.shape[:2]

    # draw and show bounding boxes
    bounding_boxes, confidences, classes = drawBoundingBoxes(output, probability_minimum,  w, h)

    # include class names by index --> similar to phase 3
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
    coco_labels = 80
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')

    list_of_classes = annotatingImages(results,bounding_boxes, colours, classes, image, labels, confidences, image_name)
    inference_time = time.time() - inference_time

    return inference_time, list_of_classes

# If the provided input is specified as a directory
def isDirectory(argv, list_inference_time, list_images_inputed, list_all_image_detections):
    list_images = os.listdir(argv[4])
    for img in list_images:
        if(img[-4:] == ".jpg"):
            # run YOLO
            inf_time, list_of_classes = runYOLO(path_to_cfg = argv[1], path_to_weights = argv[2], 
                                                      path_image= argv[4] + "/" + img, path_labels = argv[5])

            list_inference_time.append(inf_time)
            list_images_inputed.append(img)
            list_all_image_detections.append(list_of_classes)
            arg_index = 6
        else:
            print("Invalid file in directory! Use -h to get show functionality")
            break

# If the provided input is specified as multiple images
def isMultImages(argv, list_inference_time, list_images_inputed, list_all_image_detections):
    num_imgs = getNumImages(arguments = argv)
    for n in range(num_imgs):
        # run YOLO
        inf_time, list_of_classes = runYOLO(path_to_cfg = argv[1], path_to_weights = argv[2], 
                                                  path_image= argv[4+n], path_labels = argv[4+num_imgs])

        list_inference_time.append(inf_time)
        list_images_inputed.append(getImageName(argv[4+n])+".jpg")
        list_all_image_detections.append(list_of_classes)
    arg_index = 5+num_imgs

# If the provided input is specified as a none --> assume that a single image is provided
def isSingleImage(argv, list_inference_time, list_images_inputed, list_all_image_detections):
    # run YOLO
    inf_time, list_of_classes = runYOLO(path_to_cfg = argv[1], path_to_weights = argv[2], 
                                              path_image= argv[3], path_labels = argv[4])

    list_inference_time.append(inf_time) 
    list_images_inputed.append(getImageName(argv[3])+".jpg")
    list_all_image_detections.append(list_of_classes)
    arg_index = 4

# Helper function to display number of images detected for each class --> used when -tot_img_brkdwn output flag specified
def perImgCount(list_images_inputed, list_all_image_detections ):
    # Iterate through each image
    for x in range(len(list_images_inputed)):
        str_per_img_brkdwn = list_images_inputed[x] + " =>"
        dict_classes_per_image = makeListToDict(list_all_image_detections[x]) 
        # form string object containing number of class instances found within each image
        for i in dict_classes_per_image:
            if(len(dict_classes_per_image) == 1):
                str_per_img_brkdwn = str_per_img_brkdwn + " " + str(i) +" : " + str(dict_classes_per_image[i]) 
            else:
                str_per_img_brkdwn = str_per_img_brkdwn + " " + str(i) +" : " + str(dict_classes_per_image[i]) + " |"
        print(str_per_img_brkdwn)

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
    filePrint('help.txt', 1, 19)
    
    user = input("Type for more info: ")
    if user == "run":
        filePrint('help.txt', 19, 26)
    elif user == "input_flag":
        filePrint('help.txt', 26, 39)
    elif user == "output_flag":
        filePrint('help.txt', 39, 51)

# Parse for output flag and print relevant information about object detections
def outputFlag(args, list_inference_time, list_images_inputed, list_all_image_detections):
    dictAllClasses, numClassesDetected = makeListOfListToDict(list_all_image_detections)

    for x in args:
        if x == "-h":          
            printHelp()
        elif x == "-inf":        
            print(f"Average Inference: {np.mean(list_inference_time):.4f} seconds")
        elif x =="-classes_all":   
            print("Total Number of Classes Detected: ", numClassesDetected)
        elif x == "-tot_brkdwn":     
            print("Total Detection Breakdown: ")
            for i in dictAllClasses:
                print(i+":",dictAllClasses[i])
            print()
        elif x == "-tot_brkdwn_img":    
            print("Per Image Breakdown")
            perImgCount(list_images_inputed, list_all_image_detections)
        print()

# main function to specify program flow
if __name__ == '__main__':
    if(len(sys.argv)< 4):
        print("Need more arguements! Use -h to get show functionality")   
        # Assumption format is: path_to_cfg path_to_weights -image_input flag path_image path_labels -other_flags.
        
    else:
        # Variables Storing Key information:
        inference_time = []
        input_images = []
        image_detections = []
        arg_index = 0       #holds current arg_index viewing, helps with parsing flags

        # Detect input type: -d (= directory), -m (= multiple images), none (= single image)
        if (sys.argv[3] == "-d"):
            isDirectory(sys.argv, inference_time, input_images, image_detections)
        elif (sys.argv[3] == "-m"):
            isMultImages(sys.argv, inference_time, input_images, image_detections)
        else:
            isSingleImage(sys.argv, inference_time, input_images, image_detections)

        # check output flag to determine what to print: -h, -inf, -classes_all, -tot_brkdwn, -tot_brkdwn_img
        outputFlag(sys.argv[4:], inference_time, input_images, image_detections)