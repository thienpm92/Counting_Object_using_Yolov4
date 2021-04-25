# Counting_Object_using_Yolov4
This project allows users to choose whether to use the model Yolov4 or OpenCV MobileSSD for object detection and counting the number.

#Enviroment:
- numpy
- torch>=1.4.0
- opencv cuda 4.5.0
- 
#Run
python counting.py vid_name model_type height width
E.g: python counting.py test_video.mp4 yolov4 512 512
    python counting.py test_video.mp4 ssd 512 512

You can modify the object detection type and the tracking method in the code
