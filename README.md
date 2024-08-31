
<img  src="https://github.com/doguilmak/InferenceVision/blob/main/assets/Inference%20Vision%20Cover.png" alt="github.com/doguilmak/InferenceVision"/>

In contemporary scientific research and applications, there is an increasing demand for accurate geospatial analysis to address various real-world challenges, ranging from environmental monitoring to urban planning and disaster response. The ability to precisely locate and identify objects within geographic areas plays a pivotal role in such endeavors. In this scientific project, we aim to enhance geospatial analysis by integrating object detection techniques with geographic coordinate calculations. Please check our [website](https://doguilmak.github.io/InferenceVision/) for the library. You'll find a wealth of information and materials available to enrich your knowledge and learning experience.

<br>

Stay tuned for regular updates on our progress and new developments:

<details>

<summary>Latest updates...</summary>

<br>

<b>June 2024</b>
<ol>
	<li>Launched InferenceVision version 1.0!</li>
</ol>

<b>August 2024</b>
<ol>
	<li>Launched InferenceVision version 1.1!</li>
	<li>Launched InferenceVision version 1.2!</li>
</ol>

</details>

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

## **Scientific Significance**
The proposed project has several scientific implications and contributions:

- **Automation and Efficiency:** By automating the process of object detection and geographic coordinate calculation, our system significantly reduces the time and effort required for geospatial analysis, thereby enhancing efficiency and scalability.

- **Accuracy and Precision:** Through the integration of advanced algorithms, our system ensures high accuracy and precision in object detection and geographic coordinate calculation, leading to reliable and trustworthy results.

- **Versatility and Adaptability:** The developed system is versatile and adaptable to various applications, including environmental monitoring, urban planning, agriculture, and disaster response. It provides researchers and practitioners with a powerful tool for analyzing geospatial data in diverse contexts.

<br>

## **Methodology**

In this section, we outline the methodology employed for deriving geographic coordinates from input data within the InferenceVision framework. This methodological approach combines advanced techniques in satellite image analysis, object detection, and geographic coordinate calculation to enable precise geospatial analysis and visualization. Let's delve into the steps involved:

<img  src="https://github.com/doguilmak/InferenceVision/blob/main/assets/Inference%20Vision%20Intro.png" alt="github.com/doguilmak/InferenceVision"/>

**Given a set of inputs, the calculation unfolds as follows:**

<br>

**1- Transform VHR Satellite Image Coordinates to WGS 84 (EPSG:4326) and Extract Polygon Coordinates:** The target Coordinate Reference System (CRS) is WGS 84, representing a geographic coordinate system. Converting to this CRS standardizes the data. We use Nearest Neighbor interpolation, which can result in a blocky appearance. Transformed coordinates are precise to 6 decimal places. First, we transform image coordinates to WGS 84. Then, we extract polygon coordinates, defining the geographical extent with top-left and bottom-right corners as reference points for computing the geographic coordinates of normalized centers.

<br>

$$ \text{Polygon coordinates} = {(lat_{top \space left}, lon_{top \space left}), (lat_{bottom \space right}, lon_{bottom \space right})} $$

<br>

**2- Calculate Normalized Centers:** Retrieve the pixel coordinates of the bounding box, then calculate its center. Next, proceed to normalize each center of the bounding box using image size. Representing the relative positions of objects within the defined polygon, normalized center coordinates are structured as a matrix with rows $(y_{norm}, x_{norm})$ for $i = 1, 2, \ldots, number \space of \space centers$.

<br>

<br>

**3- Calculate Geographic Coordinates:** The function calculates geographic coordinates using the following equations. For each $i$ from $1$ to $number \space of \space centers$ in the image:

<br>

$$ lat = lat_{top \space left} + (lat_{bottom \space right} - lat_{top \space left}) \times y_{norm} $$

$$ lon = lon_{top \space left} + (lon_{bottom \space right} - lon_{top \space left}) \times x_{norm} $$

   
   **Where:**

   - $lat$ represents latitude.
   - $lon$ represents longitude.
   - $y_{norm}$ and $x_{norm}$ are the normalized center coordinates.
   - $lat_{top \space left}, lon_{top \space left}, lat_{bottom \space right},$ and $lon_{bottom \space right}$ are the latitude and longitude of the top-left and bottom-right corners of the polygon, respectively.

<br>

<img  src="https://github.com/doguilmak/InferenceVision/blob/main/docs/DifferenceMap.gif" alt="github.com/doguilmak/InferenceVision"/>

<br>

**NOTE: The input image must have a CRS set to ensure accurate geographic coordinate calculation.**

<br>

## **Install and Use the Library**

<br>

**1- Install the Library Run the following command in a code cell to install `inference_vision` from GitHub:**

    git clone https://github.com/doguilmak/InferenceVision.git
    cd InferenceVision


**2- Install requirements using `requirements.txt` file.**
	
    pip install -r requirements.txt -q

**3- Once the installation is complete, import the `InferenceVision` class from the library.** 

    from inference_vision import InferenceVision

**4- Here's a simple example demonstrating how to use `InferenceVision`:**

    inference = InferenceVision(
         tif_path="path/to/image.tif",
	     model_path="path/to/model.pt"
	)
    
    inference.process_image(build_csv=True, csv_filename="output.csv")

<br>

**In addition, you can see how to use the `inference_vision` library step by step in an [IPython Notebook](https://github.com/doguilmak/InferenceVision/blob/main/usage/InferenceVision_Usage.ipynb) environment.**

<br>

### **Debugging and Future Improvements**

**Debugging:** In case of any errors or unexpected behavior during image processing, carefully review the input data, model configuration, and method calls. Use debugging tools such as print statements, logging, or interactive debugging to identify and resolve issues.

<br>

**Future Improvements:** Consider incorporating additional features or enhancements to further optimize the performance and usability of the `InferenceVision` class. Potential improvements may include support for alternative object detection models, integration with other geospatial libraries, or optimization of computational efficiency.

<br>

## **Conclusion**
This calculation elucidates the process of deriving geographic coordinates from given inputs, a pivotal step within `InferenceVision` framework. It facilitates the transformation of normalized center coordinates into precise geographic coordinates, fostering accurate geospatial analysis and visualization. Geographic coordinates, namely latitude and longitude, are indispensable for pinpointing specific locations on Earth's surface. This process outlined here harmonizes normalized center coordinates, relative values within a bounding area, into a set of coordinates mappable onto a geographical map for comprehensive analysis. In conclusion, our scientific project aims to advance the field of geospatial analysis by leveraging cutting-edge technologies and methodologies. By combining object detection with geographic coordinate calculation, we strive to provide researchers and practitioners with an efficient, accurate, and versatile solution for addressing complex geospatial challenges.

<br>

## **Citation**

For a detailed exploration of related work, refer to the research article available at [ResearchGate](https://www.researchgate.net/publication/381636131_A_GEOSPATIAL_DATAFRAME_OF_COLLAPSED_BUILDINGS_IN_ANTAKYA_CITY_AFTER_THE_2023_KAHRAMANMARAS_EARTHQUAKES_USING_OBJECT_DETECTION_BASED_ON_YOLO_AND_VHR_SATELLITE_IMAGES). Presented at IEEE IGARSS 2024 in Athens, our article delves into the application of object detection techniques in geospatial contexts, highlighting the ultimate use of Very High Resolution (VHR) satellite imagery for analyzing disaster impacts.
