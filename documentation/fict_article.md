# Creating a Custom Dataset and AI Model for Predicting Sail Deformations: Integrating Computer Vision for Data Extraction and Machine Learning for Predictive Analysis

Sailboats rely heavily on the precise adjustment of sails to achieve optimal performance under varying environmental conditions. The relationship between sail input controls and the resulting sail shape is complex, nonlinear, and not yet fully understood. While computational fluid dynamics (CFD) and manual methods exist to analyze sail performance, they are time-intensive and lack real-time applicability.

Professional sailing teams require real-time sail shape prediction capabilities in their simulation environments. Machine learning presents a promising solution, potentially enabling accurate 3D sail shape predictions based on datasets that combine input controls, environmental factors, and sail shape representations.

A key challenge lies in accurately defining the sail shape and linking it to input controls and environmental factors. Sail shape is traditionally measured through costly, manual processes, making large-scale data collection impractical. Furthermore, there is a lack of datasets that integrate sail input controls, environmental conditions, and visual sail shapes for predictive modeling using advanced machine learning techniques. This presents a clear gap in the AI field where computer vision techniques and AI models can be used to offer potential soluitions for real time sail shape analysis.

This project addresses these challenges through a comprehensive data collection and analysis pipeline. The process begins with capturing sail photographs during active sailing conditions (see Fig. 1), while simultaneously recording input control settings and environmental variables. An automated computer vision pipeline was developed to extract sail splines from these images (see Fig. 2), enabling the generation of numerical representations of sail shapes. This dataset will then be used to train a neural network capable of predicting sail shapes based on input controls and environmental factors.

The computer vision pipeline processes sail images through a series of sequential operations to extract precise spline coordinates. Initially, color thresholding isolates the colored splines from the sail and background. This is followed by Gaussian blurring to reduce image noise, morphological dilation to enhance spline connectivity, and Canny edge detection to identify spline boundaries. From these processed images, we extract three key sail shape parameters: twist, camber, and draught. Twist represents the angular deviation between successive splines, calculated relative to the lowest visible spline (as the centerline is not visible in our imaging setup). Camber (C) quantifies the maximum perpendicular distance from the spline to its chord line, while draught (D) measures the distance from the leading edge to the point of maximum camber. Both camber and draught are normalized and expressed as percentages of the spline's chord length to ensure consistent measurements across different sail configurations.

This project provides compelling evidence that both core challenges - real-time sail shape prediction and automated data collection - can be effectively addressed through the integration of machine learning and computer vision techniques.The successful implementation demonstrates that complex sail dynamics can be captured and predicted using advanced ai techniques , offering a significant advancement over traditional CFD and manual approaches.

this
\

one image

2 more paragraphs in more detail about the cv pipeline and the neural network methodology.
