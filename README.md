
<img  src="https://github.com/doguilmak/InferenceVision/blob/main/assets/Inference%20Vision%20Cover.png" alt="github.com/doguilmak/InferenceVision"/>

In contemporary scientific research and applications, there is an increasing demand for accurate geospatial analysis to address various real-world challenges, ranging from environmental monitoring to urban planning and disaster response. The ability to precisely locate and identify objects within geographic areas plays a pivotal role in such endeavors. In this scientific project, we aim to enhance geospatial analysis by integrating object detection techniques with geographic coordinate calculations. Please check our [website](https://doguilmak.github.io/InferenceVision/) for the library. You'll find a wealth of information and materials available to enrich your knowledge and learning experience.

<br>

Stay tuned for regular updates on our progress and new developments:

<details>

<summary>Latest updates...</summary>

<br>

<b>June 2026</b>
<ol>
	<li>Launched InferenceVision version 1.3!</li>
</ol>

<b>August 2025</b>
<ol>
	<li>The fine-tuned GPT-Neo-1.3B InferenceVision Q&A LLM model is now available on <a href="https://huggingface.co/doguilmak/inferencevision-gpt-neo-1.3B">HuggingFace</a>.</li>
</ol>

<b>May 2025</b>
<ol>
	<li>The fine-tuned pythia-1B InferenceVision Q&A LLM model is now available on <a href="https://huggingface.co/doguilmak/inferencevision-pythia-1B">HuggingFace</a>.</li>
</ol>

<b>December 2024</b>
<ol>
	<li>Introduced the advanced language model for technical Q&A with InferenceVision!</li>
</ol>

<b>August 2024</b>
<ol>
	<li>Launched InferenceVision version 1.1!</li>
	<li>Launched InferenceVision version 1.2!</li>
</ol>

<b>June 2024</b>
<ol>
	<li>Launched InferenceVision version 1.0!</li>
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

This section describes the methodology used in the InferenceVision framework to derive geographic coordinates from object detections in Very High Resolution (VHR) satellite imagery.

<img src="https://github.com/doguilmak/InferenceVision/blob/main/assets/Inference%20Vision%20Intro.png" alt="github.com/doguilmak/InferenceVision"/>

### **Step 1: Transformation to WGS 84 (EPSG:4326)**

The image boundary is transformed from its source Coordinate Reference System (CRS) to WGS 84 (EPSG:4326), ensuring that all coordinates are expressed in geographic latitude and longitude.

The transformation produces the coordinates of the image corners:

$$
(lon_{TL}, lat_{TL}),(lon_{BR}, lat_{BR})=
\text{transform}
\left(
bounds_{dataset},
CRS_{dataset}
\rightarrow
EPSG:4326
\right)
$$

Where:

- $lon_{TL}, lat_{TL}$ = longitude and latitude of the top-left corner
- $lon_{BR}, lat_{BR}$ = longitude and latitude of the bottom-right corner

Coordinates are stored with a default precision of nine decimal places.

<br>

### **Step 2: Computation of Normalized Object Centers**

YOLO returns bounding boxes directly in center format:

$$
(x_c, y_c, w, h)=\texttt{box.xywh}
$$

Where:

- $x_c$ = x-coordinate of the bounding box center
- $y_c$ = y-coordinate of the bounding box center
- $w$ = bounding box width
- $h$ = bounding box height

The center coordinates are normalized using the image dimensions:

$$
N_x = \frac{x_c}{W}
$$

$$
N_y = \frac{y_c}{H}
$$

Where:

- $N_x$ = normalized horizontal coordinate
- $N_y$ = normalized vertical coordinate
- $W$ = image width (pixels)
- $H$ = image height (pixels)

The normalized coordinates are mapped as:

$$
N_x \rightarrow \text{Longitude}
$$

$$
N_y \rightarrow \text{Latitude}
$$

<br>

### **Step 3: Geographic Coordinate Interpolation**

The geographic coordinates of each detected object are obtained by linearly interpolating between the transformed image corners.

Longitude:

$$
lon=lon_{TL}+(lon_{BR}-lon_{TL})\cdot{N_x}
$$

Latitude:

$$
lat=lat_{TL}+(lat_{BR}-lat_{TL})\cdot{N_y}
$$

Where:

- $lon$ = longitude of the detected object
- $lat$ = latitude of the detected object
- $N_x, N_y$ = normalized center coordinates
- $lon_{TL}, lat_{TL}$ = top-left corner coordinates
- $lon_{BR}, lat_{BR}$ = bottom-right corner coordinates

The interpolation assumes a linear relationship between image coordinates and geographic coordinates over the image extent.

<img src="https://github.com/doguilmak/InferenceVision/blob/main/docs/DifferenceMap.gif" alt="github.com/doguilmak/InferenceVision"/>

**NOTE: The input image must have a CRS set to ensure accurate geographic coordinate calculation.**

<br>


## **Install and Use the Library**

### **1. Clone the Repository**

```bash
git clone https://github.com/doguilmak/InferenceVision.git
cd InferenceVision
```

### **2. Install Requirements**

```bash
pip install -r requirements.txt -q
```

### **3. Import the Library**

```python
from inference_vision import InferenceVision
```

### **4. Initialize InferenceVision**

```python
iv = InferenceVision(
    tif_path="path/to/image.tif",
    model_path="path/to/model.pt",
    coord_precision=9
)
```

#### Parameters

| Parameter | Description |
|------------|------------|
| `tif_path` | Path to the input GeoTIFF image |
| `model_path` | Path to the YOLO model weights (`.pt`) |
| `coord_precision` | Decimal precision for geographic coordinates (default = 9) |

> **Note:** The input image must contain a valid CRS definition to ensure accurate geographic coordinate calculation.

### **5. Run Inference**

#### CSV Output

```python
iv.process_image(
    output_format="csv",
    csv_filename="results.csv"
)
```

#### GeoJSON Output

```python
iv.process_image(
    output_format="geojson",
    geojson_filename="results.geojson"
)
```

#### Console Output

```python
iv.process_image()
```

### **Output Formats**

| Format | Description |
|----------|-------------|
| `csv` | Saves detections and geographic coordinates to a CSV file |
| `geojson` | Saves detections as a GeoJSON FeatureCollection |
| `None` | Prints results directly to the console |

### **Reusing the Same Instance**

The internal state is automatically reset after each call to `process_image()`, allowing the same `InferenceVision` instance to be safely reused across multiple images.

### **Notebook Tutorial**

**In addition, you can see how to use the `inference_vision` library step by step in an [IPython Notebook](https://github.com/doguilmak/InferenceVision/blob/main/usage/InferenceVision_Usage.ipynb) environment.**

<br>

## **Advanced Language Models for InferenceVision**

To enhance your experience with **InferenceVision**, we've integrated two advanced language models—[**InferenceVision-Pythia-1B**](https://huggingface.co/doguilmak/inferencevision-pythia-1B) and [**InferenceVision-GPTNeo-1.3B**](https://huggingface.co/doguilmak/inferencevision-gpt-neo-1.3B)—that deliver intelligent, context-aware support for technical topics. These models are designed to assist with questions related to geospatial analysis, object detection, and geographic coordinate calculations, helping users understand the technical foundation of the platform.

### **Model Overview**

#### [**InferenceVision-Pythia-1B**](https://huggingface.co/doguilmak/inferencevision-pythia-1B)
Based on the **EleutherAI/pythia-1b** architecture, this high-capacity model is fine-tuned specifically for the InferenceVision domain. It excels at generating detailed responses to complex questions involving object detection, spatial data handling, and coordinate systems. The model’s training was tailored to domain-specific documentation and technical prompts, ensuring precise and relevant answers.

#### [**InferenceVision-GPTNeo-1.3B**](https://huggingface.co/doguilmak/inferencevision-gpt-neo-1.3B)
Built on the **EleutherAI/gpt-neo-1.3B** model, this transformer-based language model has been fine-tuned using a structured Q&A dataset customized for InferenceVision. It offers highly accurate responses related to geospatial workflows, coordinate transformations, and detection pipelines, making it a reliable assistant for navigating technical content.

### **Key Features:**
- **Domain-Specific Tuning**: Both models are optimized using project-specific content to ensure high relevance and contextual precision.
- **Large-Scale Performance**: With 1B+ parameters, these models handle complex language tasks and detailed technical queries.
- **Q&A Optimization**: Structured to deliver targeted support for object detection, spatial analysis, and platform workflows.
- **Context-Aware Responses**: Capable of understanding and responding to intricate prompts across geospatial and AI domains.

For a hands-on guide on fine-tuning and using these models with **InferenceVision**, check out the [interactive notebook](https://github.com/doguilmak/InferenceVision/blob/main/usage/InferenceVision_LLM_QA.ipynb).

<br>

## **Conclusion**  
  
InferenceVision provides an end-to-end workflow for object detection and geospatial coordinate extraction from Very High Resolution (VHR) imagery.  
  
The framework automatically:  
  
1. Transforms image bounds to WGS 84 (EPSG:4326)  
2. Runs YOLO object detection  
3. Computes precise geographic coordinates for detected objects  
4. Exports results in CSV or GeoJSON formats  
  
By transforming image-space detections into real-world geographic coordinates, InferenceVision enables efficient spatial analysis, mapping, environmental monitoring, disaster assessment, and other geospatial applications.  
  
The framework combines modern object detection techniques with geographic coordinate interpolation, providing researchers and practitioners with a practical and scalable solution for geospatial intelligence workflows.

<br>

## **Citation**

For a detailed exploration of related work, refer to the research article available at [IEEE](https://ieeexplore.ieee.org/document/10642920). Presented at IEEE IGARSS 2024 in Athens, our article delves into the application of object detection techniques in geospatial contexts, highlighting the ultimate use of Very High Resolution (VHR) satellite imagery for analyzing disaster impacts.

<br>

**BibTeX**:

    @INPROCEEDINGS{10642920,
      author = {Ilmak, Dogu and Iban, Muzaffer Can and Zafer Şeker, Dursun},
      title = {A Geospatial Dataframe of Collapsed Buildings in Antakya City after the 2023 Kahramanmaraş Earthquakes Using Object Detection Based on YOLO and VHR Satellite Images},
      booktitle = {IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium},
      year = {2024},
      pages = {3915-3919},
      keywords = {YOLO; Buildings; Urban areas; Earthquakes; Geoscience and remote sensing; Satellite images; Sensors; Geospatial analysis; Context modeling; Deep Learning; Object Detection; Very High-Resolution Satellite Imagery; Remote Sensing; Earthquake Damage Assessment},
      doi = {10.1109/IGARSS53475.2024.10642920}
    }
