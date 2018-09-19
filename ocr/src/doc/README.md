# Detect document

Steps :
1.  Pre-process image : threshold , blur , detect edge
2.  Use cv2.houghLines to find any possible line in image
3.  Use a set of rules to filter from all possible lines above to get the 
    bounding box of document
4.  Cut the image
