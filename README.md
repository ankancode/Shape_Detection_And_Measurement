List of libraries:

	cv2
	darkflow
	matplotlib
	scipy
	imutils
	argparse
	json
	numpy 
	os



Technical Stack:

	Python
	Tensorflow (darkflow YOLO in based on it)
	cv2 (for image processing) 



Training: 

	Training of YOLOv2 was done using "tiny-yolo-voc" weights

	

Note:

	We don't need to handle the resolution of the image, as it is handled by YOLOv2 internally.

	To calculate the dimensions of the detected shape, i have calculated a factor which helps to manipulate and map the pixel length to the length in centimeters.
	This Factors depends on the screen resolution and screen size of the computer.
	For my case the screen resolution is 1366x768 and screen width is 31 cms.
	Using this i get a factor of 44.
	1366/31 = 44
	This should work for most of the scenarios.



Further Improvements:

	Future if we wish to make it resolution independent we can give back the ratio of the length of the object with respect to length of the image.



Run Command : 

	python Detection_position_measurement_service.py -m <cfg_path> -l <labels_path> -c <cpkt_path> -i <image_path>



Dataset Description:

	Total number of Images : 544

	Statistics about the labels:

	Square: 598
	Triangle: 875
	Circle: 505
	Rectangle: 741


Output:

	It will give you 3 outputs.

	1. object_measurements_position_jsonData [which will contain the following: "label","confidence","topleft(co-ordinates)", "bottomright(co-ordinates)"] from this we can obtain the label of the detected shape and the location of it

	2. object_detection_jsonData ("Object_Id", "Height", "Width") from this we can obtain the height and width of the shapes.

	3. And an output directory will be created which will contain the image with detected shapes along with their height and width measurement.