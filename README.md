Download the 1,3 GB Zip (Labelled Data) from http://vision.soic.indiana.edu/projects/egohands/ EgoHands Dataset.

Unzip it and rename the folder _LABELLED_SAMPLES into data

Run the train right in the level where the Mydataset.py and the folder data are.

It does run with os normpath so you don't have to worry about the operating system  of yours.

The first step is making the Object detection trhough the FasterRCNN's Region Proposals(First) and ROI Align(Second). The top n locations and their objectness scores are result of the / of each region proposal layer.. Compute Locations and objectness scores predictions from the RPN. I didn't used Anchors, I used Boxes, it is simpler in my perspective – not like the tutorial gives the option Approach 2. You get -rpn_cls_loss and so -rpn_reg_loss. You pass these top N locations into the Classification for the / for each ROI feeding ROI Align. You get -roi-cls-loss and -roi_reg_loss. 

The second step realizes a FCN Fully convolutional netti. Result will be the extended result as the uninvolved Faster-R-CNN, then again FCN stands not for CNN, it does not work with Feature map \ vectore map \ flattening the feature map. It does not have Fully connected layers. It doesn't retain the spatial orientation (the CNN does). However, This step works with binary masks. It will do the Pixel-To-Pixel Alignment resulting in Instance Segmentation. It does draw a b–box, as well.

The coco_eval, coco_utils, engine, get_model_instance_segmentation(function), transforms and util are files or functions from the TorchVision Object Detection Finetuning Tutorial, you may update them by time.

It works with save \ load model and model's state-dict so it can be apply on unknown picture.

This project is still on-going, for 4 hands 

data

|_CARDS_COURTYARD_B_T

|_CARDS_COURTYARD_H_S

|_CARDS_COURTYARD_S_H

|_...

your-files.py

ok Good-Bye!
