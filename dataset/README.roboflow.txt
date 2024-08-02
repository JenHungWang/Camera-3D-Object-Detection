
MA - v2 Camera_1280
==============================

This dataset was exported via roboflow.com on December 11, 2022 at 11:36 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 2154 images.
AV are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 1280x1280 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random rotation of between -7 and +7 degrees
* Random brigthness adjustment of between -10 and +10 percent
* Random exposure adjustment of between -10 and +10 percent
* Random Gaussian blur of between 0 and 0.25 pixels
* Salt and pepper noise was applied to 2 percent of pixels


