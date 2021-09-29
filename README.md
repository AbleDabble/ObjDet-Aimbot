# ObjDet-Aimbot
Uses YOLOv5 object detection to power an aimbot


## How to use
I believe including the models for the game splitgate as well as the training data I used would be a violation of splitgates terms of service and a potential copyright violation so you will have to generate your own models and training data. https://github.com/ultralytics/yolov5 for information on training.

### Training Data Outline
Training data consisted of 500 images captured from splitgate using a simple triggerbot that captures an image every 4 seconds if there is change in reticle color. Please see my colorbot for an example of how you can easily do this in splitgate. Next the data are given two labels: head and body. The body did not include any limbs (ie arms, legs or head) to ensure that the center of the bounding box always included some part of the enemy. Only enemies were labeled. 10% of the training data were  negative samples of images with just teammates that remained unlabeled to ensure that the bot did not attempt to target teammates  The data was labeled using https://www.makesense.ai/

### File structure

