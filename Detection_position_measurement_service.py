import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import json 
import os

'''
Overiding the "default" function from JSONEncoder class so that 
"TypeError" can be handled properly
'''
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

'''
Function finding the midpoint
'''
def midpoint(pointA, pointB):
	return ((pointA[0] + pointB[0]) * 0.5, (pointA[1] + pointB[1]) * 0.5)


'''
Function for loading the yolo model
'''
def model_load(model_path, labels_path, ckpt_path):
	options = {
	    'model': model_path,
	    'load': -1,
	    'threshold': 0.4,
	    'labels': labels_path,
	    'backup': ckpt_path
	}

	tfnet = TFNet(options)

	return tfnet

'''
Function for detecting the shapes, getting their positions co-ordinates
and calculating the height and width of the shapes
'''
def detection_position_measurement(model, image_path):

	# read the color image and covert to RGB
	img = cv2.imread(image_path, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# use YOLO to predict the image
	result = model.return_predict(img)

	print(img.shape)

	# convert the image to grayscale, and blur it slightly
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	# find contours in the edge map
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	 
	# sort the contours from left-to-right 
	(cnts, _) = contours.sort_contours(cnts)

	orig = img.copy()
	object_measurements=[]
	counter = 1

	
	# iterate over every object in the image and calculate
	# the height and width of each object
	for c in cnts:

	    item = {"Object_Id": counter }

	    # if the contour is not sufficiently large, ignore it
	    if cv2.contourArea(c) < 100:
	        continue

	    # compute the rotated bounding box of the contour
	    box = cv2.minAreaRect(c)
	    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	    box = np.array(box, dtype="int")

	    # order the points in the contour such that they appear
	    # in top-left, top-right, bottom-right, and bottom-left
	    # order, then draw the outline of the rotated bounding
	    # box
	    box = perspective.order_points(box)
	    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	    # loop over the original points and draw them
	    for (x, y) in box:
	        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
	        
	    # unpack the ordered bounding box, then compute the midpoint
	    # between the top-left and top-right coordinates, followed by
	    # the midpoint between bottom-left and bottom-right coordinates
	    (tl, tr, br, bl) = box
	    (tltrX, tltrY) = midpoint(tl, tr)
	    (blbrX, blbrY) = midpoint(bl, br)

	    # compute the midpoint between the top-left and top-right points,
	    # followed by the midpoint between the top-righ and bottom-right
	    (tlblX, tlblY) = midpoint(tl, bl)
	    (trbrX, trbrY) = midpoint(tr, br)

	    # draw the midpoints on the image
	    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	    # draw lines between the midpoints
	    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
	        (255, 0, 255), 2)
	    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
	        (255, 0, 255), 2)
	    
	    # compute the Euclidean distance between the midpoints
	    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	        
	    # compute the size of the object by dividing the size of the edge in pixels by the factor computed 
	    # refer to the Readme for knowing further about the factor computation
	    dimA = dA / 44
	    dimB = dB / 44

	    # draw the object sizes on the image
	    cv2.putText(orig, "{:.1f}cm".format(dimA),
	        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
	        0.65, (0,0,0), 2)
	    
	    cv2.putText(orig, "{:.1f}cm".format(dimB),
	        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
	        0.65, (0,0,0), 2)
	    
	    item["Height"] = dimA
	    item["Width"] = dimB
	    
	    object_measurements.append(item)
	    counter += 1

	# create the JSON for the measurements of the object
	object_measurements_jsonData = json.dumps(object_measurements)

	# loop over results of detection and pull out info 
	for i in range(len(result)):
	    tl = (result[i]['topleft']['x'], result[i]['topleft']['y'])
	    br = (result[i]['bottomright']['x'], result[i]['bottomright']['y'])
	    label = result[i]['label']


	    # add the box and label and display it
	    orig = cv2.rectangle(orig, tl, br, (0, 255, 0), 7)
	    orig = cv2.putText(orig, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
	

	image_name = image_path.split("/")[-1]    
	main_path = os.getcwd()
	image_dir = os.path.join(main_path,"output")
	if not os.path.exists(image_dir):
		os.makedirs(image_dir)
	detected_image_path = os.path.join(image_dir, image_name)

	# save the image with detected shapes and measured object in the output directory
	cv2.imwrite(detected_image_path,orig)

	# create the JSON for the detection results
	object_detection_position = json.dumps(result, cls=MyEncoder)

	return object_detection_position, object_measurements_jsonData


'''
    Main function, calls all the neccesary modules
'''
def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to the model cfg")
    ap.add_argument("-l", "--labels", required=True,
                    help="path to the labels.txt")
    ap.add_argument("-c", "--ckpt", required=True,
                    help="path to the ckpt directory")
    ap.add_argument("-i", "--image", required=True,
                    help="path to the ckpt directory")
    args = vars(ap.parse_args())

    # call the model_load function 
    model = model_load(args["model"], args["labels"], args["ckpt"])

    # call the detection_position_measurement function
    object_detection_position_jsonData, object_measurements_jsonData = detection_position_measurement(model, args["image"])

    print("\nobject_detection_position_jsonData : "+ object_detection_position_jsonData)
    print("\n"+"="*25)
    print("\nobject_measurements_jsonData : "+ object_measurements_jsonData)


if __name__ == "__main__":
    main()
    print("\n\nDONE !!!\n\n ")

