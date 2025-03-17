# Human Activity Detection Using Multimodal Data Documentation

## Computer Vision Pipeline

### Environment Set Up:

To run `IOU_Vector_Generation.py`, a conda environment is required to set up some dependencies. Follow these steps:

  

1. Create a new conda environment. You can name it anything, just replace **ultralytics-env** with the name you want to use:

`conda create --name ultralytics-env python=3.11 -y`

2. Activate the environment:

`conda activate ultralytics-env`

3. Check if the environment has been created properly. It should appear after running this command:

`conda env list`

4. Install the Ultralytics package:

`conda install -c conda-forge ultralytics`

5. Install pytorch, and pytorch-cuda:

`conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics`

6. Add the ffmpeg dependency

`install -c conda-forge ffmpeg`

*uses https://docs.ultralytics.com/guides/conda-quickstart/#setting-up-a-conda-environment*

  

### Process Pipeline

To run the Computer Vision Pipeline:

  

1. Choose a video and change the corresponding variables in the .py files

  

2. Run the IOU Algorithm:

* Run `IOU_Vector_Generation.py` by typing `python IOU_Vector_Generation.py` on terminal

* Change the input files path in `IOU_Activity_Detection.py` to match the files outputted by `IOU_Vector_Generation.py`. Run the code by typing `python IOU_Activity_Detection.py` on the terminal

  

3. Run Path Algorithm:

* Run `Path_Person_Vector_Generation.py` for getting the path tracking vector of the person. This keep track of the (x,y) of the person for each frame.

* Run `python3 Path_Object_Vector-Generation.py` for getting the object detection csv. This csv keep track of all object in different frames.
* Change the input files for `python3 Path_Actvity_Detection.py` the output of two previous vector generation program. Then run the program.  

4. Run Merged Algorithm:

* Before running, change the input to the output from `IOU_Activity_Detection.py` and `Path_Actvity_Detection.py`.
* Run `python3 Path_Actvity_Detection.py` to get object detection and duration using path tracking method. 

### Code

* `IOU_Vector_Generation.py` - This program takes an input video (the path can be modified using `input_video_path`variable), converts it to MP4 format if necessary, and processes it frame by frame using YOLOv11’s object detection and pose estimation models. The outputs of these models are stored in two CSV files:
1. Object detection data – Contains the frame number, detected objects, and their bounding boxes.
2. Pose data – Includes the frame number and the coordinates of key human features (e.g., nose, right ear, left ear, left knee, right shoulder, etc.).
This can be run on the terminal using `python IOU_Vector_Generation.py`. Run this program first when using the IOU Algorithm.

* `IOU_Activity_Detection.py` - This program takes in two input CSV files that are outputted as result of `IOU_Vector_Generation.py`.
Change `object_csv_file` variable to equal the object detection data CSV path and `pose_csv_file` variable to equal the pose data CSV path. It uses these two files to perform action detection using IOU (Intersection Over Union) & pose rules. It outputs the time stamps and action type in a CSV file called `actions.csv`. The format of the `actions.csv` is Action Type, Start Frame, and End Frame.

* `Path_Person_Vector_Generation.py`: This program takes in a video and outputs the path tracking of the person. It contain ['frame', 'x', 'y'] and for each frame for each frame as a .csv file.

* `Path_Object_Vector-Generation.py`: This program takes in a video and output object with more than 70% confidence level in the video. It outputs ["frame", "object"] for each frame as a .csv file.

* `Path_Actvity_Detection.py`: This program takes in both .csv files. It segments the video into segmentation. Then output the activity detected for each segment and their start/end time.

* `Merge_Combinging Result.py` to combine the result from path tracking and IOU algorithm. This file takes two .csv result data and combines the result together. It output a .csv file containing ["Action", "Start Time", "End Time"].

### Dataset

Videos are located on the shared Google Drive, as they are too big to upload to the Git

  

## Vibrational Pipeline

### Process Pipeline

1. Plug hard drive into the server (Raspberry Pi)

2. Plug power into server

3. Wait until the “square pi” wifi hotspot shows up. Log into the hotspot.

5. Go to this website http://192.168.4.1.8000/ where the state of each sensor can be monitored.

6. Once data is collected, run the `majority_vote_classifier_output.py` script to extract the desired data in the csv formats to train the model.

  

### Code

*  `1d_cnn_classifier.ipynb`: Program to train 1D CNN, utilized to train each activity dataset separately. To run the program, download the file, upload the corresponding datasets for a single activity, and run the Jupyter Notebook. To train multiple activities, reupload the corresponding datasets for another activity, alter paths when loading the data, and run the Notebook again.
              * When testing, this program was run through `Google Colab` as a GPU would speed up training and running time. Additionally, both students working on this program were unable to log into the Library VPN due to an unknown error, thus were unable to connect to the class server to utilize the provided GPUs.

*  `majority_vote_classifier_output.py`: The program to collect the amplitudes of the vibrational propagations and their associated binary, ground-truth labels (0-no activity, 1-activity occurring) based on a specific window size that can be set in the code. The code utilizes an array of timestamps in which the activity of interest has occurred, collects that data from the hard drive, and utilizes a majority voting algorithm to assign each window of data to a binary label. The code outputs two csv files: window_amplitude.csv and binary_output.csv, where each row of both csv files represent a window of data.

*  `save_data_csv.py`: This program collects the vibrational amplitude data stored in the numpy files of the hard drive and puts them in a csv file outputted by the program.

### Dataset

* `compiled_vibration_data_train`: Folder that contains the file output of the `majority_vote_classifier_output.py` script.

## Credits

Lana Wong, Jinsi Guo, Zainah Masood and Fatima Shaik, Yui Ishihara, and Chen-Nee Chuah
