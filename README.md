# Kneron yolov3 inference
This instruction allows to convert yolov3 from darknet to nef and evaluate mAP after convertation. Given notebook is based on Yolo example from http://doc.kneron.com/docs/#toolchain/yolo_example/ .
## Installation
```
# Download docker and run:
docker pull kneron/toolchain:latest
docker run -p 8888:8888 --rm -it -v /mnt/docker:/docker_mount kneron/toolchain:720
# In docker use jupyter notebook to get to the workspace:
pip install notebook
git clone https://github.com/SashaAlderson/Kneron_yolov3_inference
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root 
```
Use notebook to proceed further. First steps are copied from Yolo example, but we made changes to E2E simulator files, so that you can run multiple models in the same time, allowing you to use script with multithreading.
# Parallel.py
Parallel.py is a versatile script that allows you to run demo or inference on multiple models. 
Run help to check arguments:
```
python /data1/parallel.py -h 

usage: parallel.py [-h] [--demo] [--image IMAGE] [--path PATH] [--nef NEF]
[--step STEP] [--init INIT] [--model MODEL]
[--threads THREADS] [--img-size IMG_SIZE] [--conf-t CONF_T]
[--iou-t IOU_T]

Runs an inference on multiple images.

optional arguments:
-h, --help           show this help message and exit
--demo               run demo on your image
--image IMAGE        path to your image for demo
--path PATH          directory of your images
--anchors ANCHORS    path to your anchors
--nef NEF            path to your nef model
--step STEP          number of images for one model in every step
--init INIT          initialization time between models
--model MODEL        model's number
--threads THREADS    choose number of workers for inference(only for 520 model)
--img-size IMG_SIZE  image size
--conf-t CONF_T      confidence threshold
--iou-t IOU_T        iou threshold for NMS
```
## Run Demo
```
python /workspace/Kneron_yolov3_inference/parallel.py --demo --image <path_to_your_image>
```
## Run inference

```
# supposed to be used in a loop
python /workspace/Kneron_yolov3_inference/parallel.py ???path /workspace/COCO/val2014 --model 0 --step 5 --init 0 --conf-t 0.001 --threads 16 # run inference on ???0??? model with 16 threads and 5 images per iteration of script
```
Use code from notebook to run this script on COCO images with multiple models until all predictions are made.
# mAP
When all procedures in notebook are done, use https://github.com/rafaelpadilla/review_object_detection_metrics  to evaluate mAP. All required files located under /workspace/COCO/val2014.

## Yolov3
|        model      | mAP @<br>IoU=0.5:0.95  |  mAP @<br>IoU=0.5 |  
| :---------------: | :--------------------: | :----------------:|
|   Yolov3-520      | 0.254                  | 0.491             | 
|   Yolov3-720      | 0.252                  | 0.489             | 
| pjreddie's YOLOv3 | 0.310                  | 0.553             |
## Yolov3-tiny
|            model       | mAP @<br>IoU=0.5:0.95  |  mAP @<br>IoU=0.5  |  
| :--------------------: | :--------------------: | :----------------: |
|   Yolov3-tiny-520      | 0.114                  | 0.262              | 
|   Yolov3-tiny-720      | 0.110                  | 0.252              | 
| pjreddie's YOLOv3-tiny | 0.144                  | 0.325              | 
# Conclusion
Convertation slightly decreases mAP50 on Yolov3 by 10% and mAP50..95 by 18%. Converted yolov3-tiny loses 20% on mAP50 and mAP50..95. We observe difference between confidence on predictions of models for Kneron KL520 and Kneron KL720 due to different convertation to nef procedures. So models for KL520 showing a little bit better results than models for KL720.
