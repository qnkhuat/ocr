# OCR system
## Objective:
Extract text from image of patient's profiles taken by phone.

## Todo:
- [x] Detect the document from taken image
    - [x] Detect the edges of the document
      * 2 technics to use:
        * cv2.findContours
        * cv2.houghLines - Doing better
    - [x] Find the best rectangle fit the document from edges
      * Propose solution : Find all the intersection points, then put some constrains to get the document
- [ ] Compute offset of the locations that need to extract text from.
    - [x] Rescale the output doc to A4 size
    - [ ] Compute offset of the form
- [ ] Classify check/uncheck of checkbox in patient's profile
- [x] Using ML to detect handwriting numbers
  * Using ML trained on MNIST


## Problem explain:
Given an image of a patient profile, we need to extract some needed information like: names,  ages , whether the patient has cancer or not...
It's like a scan app.

But : We don't have to recognize all the text inside the profile because the profile comes in form which almost are titles, So we just need to extract some pre-defined locations.
### Difficulties
What hard here is that it has many handwriting text and doctors are really bad at writing.


## Worth reading:
Detect document from dropbox : http://bit.ly/2LCfXsQ

End-to-end scan system from dropbox : http://bit.ly/2LASOHn

Detect rectangle : http://bit.ly/2wMN9rq 
