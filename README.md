# Online and Real-Time Mask Monitoring
***AIPI 540 Final Project - Haoran Cai***

## Introduction
#### Backgroud

![image](https://user-images.githubusercontent.com/90671685/165628271-48b84186-9a22-4e67-bdf3-5c2de32bb0f6.png)
 - New variants of the virus are expected to occur.
 - The Omicron variant causes more infections and spreads faster than the original virus that causes COVID-19.
 - Wearing mask can help prevent severe illness and reduce the potential for strain on the healthcare system


#### Application
<img width="318" alt="image" src="https://user-images.githubusercontent.com/90671685/165628320-f35038fe-7362-4235-9640-339852c6d1f5.png">

This Application could:
 - Online Real-Time Mask Monitoring
 - Detect whether people wear masks
 - Detect the incorrect use of a face mask (especially wearing the mask with the nose out)

## Deploy the application
First play download my trained weigh.
https://drive.google.com/file/d/1Op9yqw5QzQyygrVm8Lbuzypig_leTdnA/view?usp=sharing

Please put this weights file under /models/darknet/weights

### Deployment on Local Computer
Please run the following command line:
```
pipenv shell
make run
```

### Deployment on Cloud Platform 
Please run the following command line:
```
pipenv shell
make gcloud-deploy
```

## Train your own model
The training related code is under scripts file
Please follow this step: 
1. Download the structure of the nerual network
```
wget https://pjreddie.com/media/files/darknet53.conv.74
```
2. Download the pre-trained weights
```
!wget https://pjreddie.com/media/files/yolov3.weights
```
3. Run the make file
```
make
```
4. Train the model
```
./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74
```
## Non Deep Learning Model
The non deep learning model is in /models/Non_DL.py
