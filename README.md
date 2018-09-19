## End-to-end WORKFLOW:
Input : an image capture a document

Output : The information from the document
1.  Detect the document inside image and cut it
2.  Use the config file to cut the image parts contain informations we need
3.  Use the checkbox,digit detection model to inference the cut images above

### How to use : 
  #### Make sure to have config file
      (See the config file instruction below)
  
  - Detect a single image
  ```
  python endtoend.py -i ../resources/images/test/full/img1.jpg 
  ```
  - Detect multiple images inside folder
  ```
  python endtoend.py -p ../resources/images/test/full
  ```
  - Advanced config ( use python inference.py --help to get more concrete explanation ):
  
      Change the checkbox detection model
      ```
      --checkbox ../resources/models/checkbox/lastest/model.ckpt 
      ```

      Change the digits detection model
      ```
      --digit ../resources/models/digit/model01_99.61/model.ckpt 
      ```

      Change the config file 
      ```
      -c ../resources/configs/config.conf
      ```

      Draw the target box to output
      ```
      --draw
      ```

### What does config file for ?

For any information we need to extract inside the document,we have to define it in the config file for the algorithm to know where to cut the right information images

### Building config file:

  Open a scanned document image ( can use ocr/scripts/detect.py to scan image) by any app that has coordinate at pointer
  
  Open ocr/build_config_file.py 
  
  With each area need to extract information create a bc.process_config_data inside variable data
  
    4 params need to provide:
      - image_shape (np.array/list): the shape of input image
          ex : (2000,1000)
      - target_name (str): the name of target(will be the key for information in the output)
          ex : 'ngay sinh'
      - target_bbox (str): Use the pointer to find upper left , bottom right x,y of the rectangle 
        around the area need to detect 
      ( do not set it too close to the area see resources/images/example/config_example.jpg to reference )
          ex : '172,55,256,135'
      * 1 in 2 is optinal:
      - is_checkbox (bool): if this is a check box, set this to True otherwise False
      - number_of_box (int): if not checkbox so how many numbers in it, if checkbox : don't need to input
      
      
## FURTHER IMPROVEMENT
### DETECT DOCUMENT:
Problem:
  - The detection algorithm has many constraints.
  
Todo : 
  - Refine the filter algorithms(the step after do houghlines) to get more accurate detection
  - Make the resize and transform image after detected more realistic

### CUT INFORMATION IMAGES:
Problem: 
  - The way that are using will be wrong if the document hasn't detected accurately or isn't flat when capture because we the algorithm rely on the edges of document to find the position of information on image.
  - The algorithm to detect number will be wrong if has noise in image or the rectangle around number wasn't detect correctly.

Todo:
  - Redesign the filter algorithms
  - Use cv2.houghlines to detect numbers box instead of cv2.findcontours
  - Try pixel based(see docs/) approach to get the bbox of number inside numbers box

### CHECKBOX CLASSIFICATION:
Problem:
  - Using a deep learning approach
  - The model is using 3 CNN layers( maybe 2 layers is enough)
  - The data is small and not accurate
  
Todo:
  - Design new method to generate data, collect more data
  - Try findcontours and other method in opencv to classify  

### DIGITS RECOGNIZATIONS:
Problem:
  - Accuracy is low
  
Todo:
  - Collect more data and retrain

