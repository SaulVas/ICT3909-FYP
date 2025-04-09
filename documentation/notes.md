
Typically on water sail images are analysed and 3 key numeric features are extracted, twist, camber and draught. these values are extracted from colorful splines along the sail.

Twist is the anlge at which the sail opens up as the height increases measured from either the centerline or the lowest spline. Camber is the distance of maximum depth along the spline. Draught is the distance from thecamber point (point of maximum depth) to the front of the spline. both camber and draught are expressed as a percenteage of the shortest distance from the start of the spline to the end.

The computer vision pipeline developed to extract the location of these colorful splines along the sail makes use of the following techniques:
color gradients, blurring for noise reduction, dilation and edge detection techniques.

Once the twist camber and draught data has been extracted from all the images, a
