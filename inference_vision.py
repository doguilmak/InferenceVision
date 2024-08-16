from ultralytics import YOLO
import rasterio.warp
from rasterio import features
import numpy as np
import pandas as pd

class InferenceVision:
    
    VERSION = "1.1"

    def __init__(self, tif_path, model_path):
        """
        Initialize an InferenceVision instance.

        Parameters
        ----------
        tif_path : str
            The file path to the TIFF image to be processed.
        model_path : str
            The file path to the YOLO model weights.

        Returns
        -------
        None
            This method initializes the object and does not return any value.

        Example
        -------
        Create an instance of InferenceVision
        >>> inference = InferenceVision(
                tif_path="path/to/image.tif",
                model_path="path/to/model.pt"
            )
        """
        self.tif_path = tif_path
        self.model_path = model_path
        self.extracted_polygons_lonlat = []
        self.results = []
        self.center_coords = []
        self.extracted_polygons = []

        with rasterio.open(self.tif_path) as dataset:
            self.image_width = dataset.width
            self.image_height = dataset.height

    def normalize_center_points(self, center_points):
        """
        Normalize center points relative to the image dimensions.

        Parameters
        ----------
        center_points : list of tuples
            A list of tuples where each tuple contains (x_center, y_center) coordinates.

        Returns
        -------
        list of lists
            A list of normalized coordinates where each coordinate is [normalized_x, normalized_y].

        Example
        -------
        Given center points [(50, 100), (150, 200)]
        >>> normalized_points = inference.normalize_center_points([(50, 100), (150, 200)])
        normalized_points will be [[0.048828125, 0.12962962962962962], [0.146484375, 0.25925925925925924]]
        """
        normalized_points = []
        for x_center, y_center in center_points:
            normalized_x = x_center / self.image_width
            normalized_y = y_center / self.image_height
            normalized_points.append([normalized_x, normalized_y])
        return normalized_points

    def calculate_bbox_center(self, coordinates):
        """
        Calculate the center point of a bounding box.

        Parameters
        ----------
        coordinates : list
            A list of four values representing the bounding box coordinates [x_min, y_min, x_max, y_max].

        Returns
        -------
        tuple
            A tuple (x_center, y_center) representing the center of the bounding box.

        Example
        -------
        Given bounding box coordinates [10, 20, 30, 40]
        >>> center = inference.calculate_bbox_center([10, 20, 30, 40])
        Center will be (20.0, 30.0)
        """
        x_min, y_min, x_max, y_max = coordinates
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + (width / 2)
        y_center = y_min + (height / 2)
        return x_center, y_center

    def process_image(self, build_csv=False, csv_filename=None):
        """
        Process the image using the YOLO model and optionally save the results to a CSV file.

        Parameters
        ----------
        build_csv : bool, optional
            Whether to build and save the CSV file with results. If True, the CSV file
            will be saved with the name provided in `csv_filename`. Default is False.
        csv_filename : str, optional
            The name of the CSV file to save results, if `build_csv` is True.

        Returns
        -------
        None
            If `build_csv` is True, saves results to the specified CSV file. If False,
            prints results to the console.

        Example
        -------
        Process the image and save results to a CSV file
        >>> inference.process_image(build_csv=True, csv_filename="output.csv")

        Process the image and print results to the console
        >>> inference.process_image(build_csv=False)
        """
        model = YOLO(self.model_path)

        results = model.predict(self.tif_path)

        with rasterio.open(self.tif_path) as dataset:
            mask = dataset.dataset_mask()
            for geom, val in rasterio.features.shapes(mask, transform=dataset.transform):
                geom = rasterio.warp.transform_geom(dataset.crs, 'EPSG:4326', geom, precision=6)
                self.extracted_polygons.append(geom['coordinates'][0])

            self.extracted_polygons[0] = self.extracted_polygons[0][:-1]
            self.extracted_polygons_lonlat = self.extracted_polygons[0]

            result = results[0]
            detected_objects = len(result.boxes)

            for i in range(detected_objects):
                box = result.boxes[i]
                coordinates = box.xyxy[0].tolist()
                x_center, y_center = self.calculate_bbox_center(coordinates)
                self.center_coords.append([x_center, y_center])

            normalized_points = self.normalize_center_points(self.center_coords)

            lat_top_left = self.extracted_polygons[0][0][0]
            lon_top_left = self.extracted_polygons[0][0][1]
            lat_bottom_right = self.extracted_polygons[0][2][0]
            lon_bottom_right = self.extracted_polygons[0][2][1]

            normalized_coords = np.array(normalized_points)
            geographic_coords = np.empty_like(normalized_coords)
            for i in range(normalized_coords.shape[0]):
                y_norm, x_norm = normalized_coords[i]
                lat = lat_top_left + (lat_bottom_right - lat_top_left) * y_norm
                lon = lon_top_left + (lon_bottom_right - lon_top_left) * x_norm
                geographic_coords[i] = [lat, lon]

            formatted_geographic_coords = np.array([
                [f'{lat:.9f}', f'{lon:.9f}'] for lat, lon in geographic_coords
            ])

            if build_csv:
                data_list = []
                for i in range(detected_objects):
                    box = result.boxes[i]
                    class_id = result.names[box.cls[0].item()]
                    coordinates = box.xyxy[0].tolist()
                    x_center, y_center = self.calculate_bbox_center(coordinates)
                    normalized_point = normalized_points[i]

                    data_list.append({
                        "Image": self.tif_path,
                        "Point": i,
                        "Latitude": formatted_geographic_coords[i][1],
                        "Longitude": formatted_geographic_coords[i][0],
                        "Object Type": class_id,
                        "Coordinates": box.xyxy[0].tolist(),
                        "Probability": box.conf[0].item(),
                        "Bounding Box Center": [x_center, y_center],
                        "Normalized Bounding Box Center": normalized_point
                    })

                df = pd.DataFrame(data_list)
                df.to_csv(csv_filename, index=False)
                print(f"\n\nDataFrame saved as {csv_filename}")

            else:
                for i in range(detected_objects):
                    box = result.boxes[i]
                    print(f"\nPoint {i} ", "-" * 40)
                    class_id = result.names[box.cls[0].item()]
                    print(f"Latitude: {formatted_geographic_coords[i][1]} - Longitude: {formatted_geographic_coords[i][0]}")
                    print("Object type:", class_id)
                    print("Coordinates:", box.xyxy[0].tolist())
                    print("Probability:", box.conf[0].item())
                    print("Bounding Box Center:", [x_center, y_center])
                    print("Normalized Bounding Box Center:", [normalized_points[i][0], normalized_points[i][1]])
