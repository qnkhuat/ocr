# Code about cutting images

## NOTATIONS
-   Target box: the box we need to draw
-   Anchor word : the word we select to be the anchor in order to find the target box

## Build config file:

What does config file for ?

For any information we need to extract inside the document,we have to define it in the config file for the algorithm able to cut the right images

## Cut pre-defined positions in config

We need to cut some part like : checkbox, a row of numbers... inside image

To do this : We use a model image to predifine the target bbox then produce a correspond config file for that type of document.

We use a model image and input the boundingbox of each targetbox in the model image,

The Algorithm will automatically produce a config file for this image.

Then use this config to cut the part when inference


## Number detection
After cut the part number image using config file, we need to detect the numbers inside to fit into mnist model

To do this we use cv2.findContours() and a set of rules to detect (see detect_and_cut_number of detect_and_cut_number_look_once in number_detection.py)


