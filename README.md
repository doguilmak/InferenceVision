<img  src="https://github.com/doguilmak/InferenceVision/blob/main/assets/Inference%20Vision%20Cover.png"  width=1000  height=250  alt="github.com/doguilmak/InferenceVision"/>

In contemporary scientific research and applications, there is an increasing demand for accurate geospatial analysis to address various real-world challenges, ranging from environmental monitoring to urban planning and disaster response. The ability to precisely locate and identify objects within geographic areas plays a pivotal role in such endeavors. In this scientific project, we aim to enhance geospatial analysis by integrating object detection techniques with geographic coordinate calculations.

<br>

## **Problem Statement**
Traditional methods of geospatial analysis often rely on manual identification and mapping of objects within geographical regions. However, these methods are time-consuming, labor-intensive, and prone to errors. Moreover, they may lack the scalability required for large-scale analyses. Therefore, there is a need for automated solutions that can accurately detect and locate objects within geographic areas, enabling efficient and scalable geospatial analysis.

<br>

## **Project Objective**
Our project seeks to address the aforementioned challenges by developing an automated system that combines object detection algorithms with geographic coordinate calculations. By integrating these components, we aim to achieve the following objectives:

1. **Object Detection:** Utilize state-of-the-art object detection algorithms, such as YOLO (You Only Look Once), to automatically identify and localize objects within satellite or aerial imagery.

2. **Geographic Coordinate Calculation:** Develop algorithms to calculate the geographic coordinates (latitude and longitude) of detected objects relative to a given bounding polygon. This involves converting normalized center coordinates of objects within the bounding polygon to precise geographic coordinates.

3. **Integration and Visualization:** Integrate object detection results with calculated geographic coordinates to create a comprehensive geospatial dataset. Visualize the detected objects and their geographic locations on maps for further analysis and interpretation.

<br>

## **Methodology**

In this section, we outline the methodology employed for deriving geographic coordinates from input data within the InferenceVision framework. This methodological approach combines advanced techniques in satellite image analysis, object detection, and geographic coordinate calculation to enable precise geospatial analysis and visualization. Let's delve into the steps involved:

<img  src="https://github.com/doguilmak/InferenceVision/blob/main/assets/Inference%20Vision%20Intro.png" alt="github.com/doguilmak/InferenceVision"/>

**Given a set of inputs, the calculation unfolds as follows:**

<br>

**1- Extract Polygon Coordinates from Very High-Resolution (VHR) Satellite Image:** The corner coordinates of the polygon delineate the geographical extent of interest, encompassing the top-left and bottom-right corners, serving as pivotal reference points for computing the geographic coordinates of normalized centers.

<br>

$$ \text{Polygon coordinates} = {(lat_{top \space left}, lon_{top \space left}), (lat_{bottom \space right}, lon_{bottom \space right})} $$

<br>

**2- Calculate Normalized Centers:** Retrieve the pixel coordinates of the bounding box, then calculate its center. Next, proceed to normalize each center of the bounding box using image size. Representing the relative positions of objects within the defined polygon, normalized center coordinates are structured as a matrix with rows $(y_{norm}, x_{norm})$ for $i = 1, 2, \ldots, number \space of \space centers$. The function calculates geographic coordinates using the following equations. For each $i$ from $1$ to $number \space of \space centers$ in the image:

<br>

$$ lat = lat_{top \space left} + (lat_{bottom \space right} - lat_{top \space left}) \times y_{norm} $$

$$ lon = lon_{top \space left} + (lon_{bottom \space right} - lon_{top \space left}) \times x_{norm} $$

   
   **Where:**

   - $lat$ represents latitude.
   - $lon$ represents longitude.
   - $y_{norm}$ and $x_{norm}$ are the normalized center coordinates.
   - $lat_{top \space left}, lon_{top \space left}, lat_{bottom \space right},$ and $lon_{bottom \space right}$ are the latitude and longitude of the top-left and bottom-right corners of the polygon, respectively.

**NOTE: The input image must have a coordinate reference system (CRS) set to ensure accurate geographic coordinate calculation.**

<br>

## **Scientific Significance**
The proposed project has several scientific implications and contributions:

- **Automation and Efficiency:** By automating the process of object detection and geographic coordinate calculation, our system significantly reduces the time and effort required for geospatial analysis, thereby enhancing efficiency and scalability.

- **Accuracy and Precision:** Through the integration of advanced algorithms, our system ensures high accuracy and precision in object detection and geographic coordinate calculation, leading to reliable and trustworthy results.

- **Versatility and Adaptability:** The developed system is versatile and adaptable to various applications, including environmental monitoring, urban planning, agriculture, and disaster response. It provides researchers and practitioners with a powerful tool for analyzing geospatial data in diverse contexts.

<br>

### Install and Use the Library

<br>

**1- Install the Library Run the following command in a code cell to install `inference_vision` from GitHub:**

    !git https://github.com/username/inference_vision.git



**2- Optionally, you can download rasterio and ultralytics.**

    !pip install ultralytics -q
    !pip install rasterio -q

**3- Once the installation is complete, import the `InferenceVision` class from the library.** 

    from inference_vision import InferenceVision

**4- Here's a simple example demonstrating how to use `InferenceVision`:**

    inference = InferenceVision(tif_path="path/to/image.tif",
	     model_path="path/to/model.pt", 
	     image_width=width, 
	     image_height=height) # Process image 
    
    inference.process_image(build_csv=True, csv_filename="output.csv")

<br>

**In addition, you can see how to use the `inference_vision` library step by step in an [IPython Notebook](https://github.com/doguilmak/InferenceVision/blob/main/usage/InferenceVision_Usage.ipynb) environment.**

<br>

## **Conclusion**
This calculation elucidates the process of deriving geographic coordinates from given inputs, a pivotal step within `InferenceVision` framework. It facilitates the transformation of normalized center coordinates into precise geographic coordinates, fostering accurate geospatial analysis and visualization. Geographic coordinates, namely latitude and longitude, are indispensable for pinpointing specific locations on Earth's surface. This process outlined here harmonizes normalized center coordinates, relative values within a bounding area, into a set of coordinates mappable onto a geographical map for comprehensive analysis. In conclusion, our scientific project aims to advance the field of geospatial analysis by leveraging cutting-edge technologies and methodologies. By combining object detection with geographic coordinate calculation, we strive to provide researchers and practitioners with an efficient, accurate, and versatile solution for addressing complex geospatial challenges.
