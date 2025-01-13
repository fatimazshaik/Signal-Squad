
# Signal Squad

## YOLOv11 Code:
### File Information
The file, imageYOLOv11.py, uses YOLOv11 to take in an input image and create an output image that identifies all the objects in that image. To test this code, use the /imgs folder as sample input images!

The file, videoYOLOv11.py, does the same thing but for videos. To test this code, use the videos from the Google Drive!
To change the input and ouput file path, use the local variables in the respective files.

### Setting an local pyhton environment ro run YOLOv11 
 *uses https://docs.ultralytics.com/guides/conda-quickstart/#setting-up-a-conda-environment*
 1. Create a new conda enviornment. You can name it anything, just replace **ultralytics-env** with the name u want to use:

    `conda create --name ultralytics-env python=3.11 -y`
2. Activate the environment:

    `conda activate ultralytics-env`
3. Check if environemnt has been created properly. It show appear after running this command:

    `conda env list`
4.  Install the Ultralytics package:
    
    `conda install -c conda-forge ultralytics`
5.  Install pytorch, and pytorch-cuda:

    `conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics`
6. Run the python files by:

    `python videoYOLOv11.py` or `python imageYOLOv11.py`


## Video Dataset:
Videos are located on the shared google drive as they are too big to upload on the Git>

## YOLOv3 Code:
YOLOv3 detection code is located in yolov3 code folder. Please note that weight file isn't uplaoded on git as it is too large!

## Other Notes:
hi!

## Credits
Lana Wong, Jinsi Guo, Zainah Masood and Fatima Shaik, Kartik Patwari, Yui Ishihara, Jeff Lai, and Chen-Nee Chuah
