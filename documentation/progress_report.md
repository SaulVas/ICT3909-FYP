# Predicting a Custom Dataset and AI Model for Predicting Sail Deformations: Integrating Computer Vision for Data Extraction and Machine Learning for Predictive Analysis

## Problem Definition

Sailboats rely heavily on the precise adjustment of sails to achieve optimal performance under varying environmental conditions. The relationship between sail input controls and the resulting sail shape is complex, nonlinear, and not yet fully understood. While computational fluid dynamics (CFD) and manual methods exist to analyze sail performance, they are time-intensive and lack real-time applicability.

A key challenge lies in accurately defining the sail shape and linking it to input controls and environmental factors. Sail shape is traditionally measured through costly, manual processes, making large-scale data collection impractical. Furthermore, there is a lack of datasets that integrate sail input controls, environmental conditions, and visual sail shapes for predictive modeling using advanced machine learning techniques.

This project seeks to address this gap by creating a dataset from scratch and developing an automated computer vision pipeline to extract precise sail shape parameters. The dataset will then be used to train a neural network capable of predicting sail shapes based on input controls and environmental factors.

## Aims and Objectives

### Aims:

The primary aim of this project is to create a comprehensive dataset linking sail input controls, environmental conditions, and sail shapes - extracted through an automated computer vision pipeline, and to develop a predictive model capable of estimating sail shape characteristics using machine learning techniques.

### Objectives:

1. **Dataset Creation:**

   - Capture high-resolution images of the sails under controlled conditions, ensuring alignment with input controls and environmental variables.
   - Define the sail shape using color-coded splines, translating visual data into an appropriate numerical definition.

2. **Computer Vision Pipeline:**

   - Research and implement computer vision techniques to process images and extract sail shape information accurately.
   - Create a computer vision pipeline to automate the identification and quantification of splines and their corresponding parameters from images.

3. **Model Development:**

   - Prepare the processed dataset for machine learning by ensuring it is clean, consistent, and well-labeled.
   - Train and evaluate a neural network to predict sail shape definitions based on input controls and environmental data.

4. **Validation and Testing:**

   - Test the trained neural network against unseen data to evaluate performance and applicability.

5. **Documentation and Reporting:**
   - Maintain detailed documentation of methods, data collection processes, and outcomes.
   - Present findings in a clear, accessible format for stakeholders in both the academic and sailing communities.
