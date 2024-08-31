import rasterio.warp
from rasterio import features
import numpy as np
import pandas as pd
from ultralytics import YOLO

class InferenceVision:
    
    VERSION = "1.2"

    def __init__(self, tif_path, model_path, coord_precision=9):
        """
        Initialize an InferenceVision instance with configurable precision.

        Parameters
        ----------
        tif_path : str
            The file path to the TIFF image to be processed.
        model_path : str
            The file path to the YOLO model weights.
        coord_precision : int, optional
            The number of decimal places to use for geographic coordinates. Default is 9.

        Returns
        -------
        None
            This method initializes the object and does not return any value.

        Example
        -------
        >>> iv = InferenceVision("image.tif", "model.pt", coord_precision=6)
        """
        self.tif_path = tif_path
        self.model_path = model_path
        self.coord_precision = coord_precision
        self.extracted_polygons_lonlat = []
        self.results = []
        self.center_coords = []
        self.extracted_polygons = []

        try:
            with rasterio.open(self.tif_path) as dataset:
                self.image_width = dataset.width
                self.image_height = dataset.height
        except rasterio.errors.RasterioIOError as e:
            raise ValueError(f"Error opening TIFF file {self.tif_path}: {e}")

    def normalize_center_points(self, center_points):
        """
        Normalize the center points based on image dimensions.

        Parameters
        ----------
        center_points : list of tuples
            A list of (x, y) center coordinates.

        Returns
        -------
        list of tuples
            A list of normalized (x, y) center coordinates.

        Example
        -------
        >>> iv = InferenceVision("image.tif", "model.pt")
        >>> normalized_points = iv.normalize_center_points([(100, 200), (150, 250)])
        >>> print(normalized_points)
        """
        normalized_points = []
        for x_center, y_center in center_points:
            normalized_x = x_center / self.image_width
            normalized_y = y_center / self.image_height
            normalized_points.append([normalized_x, normalized_y])
        return normalized_points

    def calculate_bbox_center(self, coordinates):
        """
        Calculate the center of a bounding box.

        Parameters
        ----------
        coordinates : list of float
            A list of four values [x_min, y_min, x_max, y_max] representing the bounding box.

        Returns
        -------
        tuple
            The (x_center, y_center) of the bounding box.

        Example
        -------
        >>> iv = InferenceVision("image.tif", "model.pt")
        >>> center = iv.calculate_bbox_center([50, 50, 150, 150])
        >>> print(center)
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
            will be saved with the name provided in csv_filename. Default is False.
        csv_filename : str, optional
            The name of the CSV file to save results, if build_csv is True.

        Returns
        -------
        None
            If build_csv is True, saves results to the specified CSV file. If False,
            prints results.

        Example
        -------
        Process the image and save results to a CSV file:
        >>> iv = InferenceVision("image.tif", "model.pt")
        >>> iv.process_image(build_csv=True, csv_filename="results.csv")

        Process the image and print results to the console:
        >>> iv = InferenceVision("image.tif", "model.pt")
        >>> iv.process_image()
        """
        try:
            model = YOLO(self.model_path)
            results = model.predict(self.tif_path)
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {e}")

        try:
            with rasterio.open(self.tif_path) as dataset:
                mask = dataset.dataset_mask()
                for geom, val in rasterio.features.shapes(mask, transform=dataset.transform):
                    geom = rasterio.warp.transform_geom(
                        dataset.crs, 'EPSG:4326', geom, precision=self.coord_precision)
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
                    [f'{lat:.{self.coord_precision}f}', f'{lon:.{self.coord_precision}f}'] 
                    for lat, lon in geographic_coords
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
                            "Confidence Score": box.conf[0].item(),
                            "Bounding Box Center": [x_center, y_center],
                            "Normalized Bounding Box Center": normalized_point
                        })

                    df = pd.DataFrame(data_list)
                    df.to_csv(csv_filename, index=False)
                    print(f"\nDataFrame saved as {csv_filename}")
                else:
                    for i in range(detected_objects):
                        box = result.boxes[i]
                        class_id = result.names[box.cls[0].item()]
                      
                        print(f"\nPoint {i} {'-' * 20}")
                        print(f"Latitude: {formatted_geographic_coords[i][1]} | Longitude: {formatted_geographic_coords[i][0]}")
                        print(f"Object Type: {class_id}")
                        print(f"Coordinates (Bounding Box): {box.xyxy[0].tolist()}")
                        print(f"Confidence Score: {box.conf[0].item():.4f}")
                        print(f"Bounding Box Center (X, Y): {self.calculate_bbox_center(box.xyxy[0].tolist())}")
                        print(f"Normalized Bounding Box Center (X, Y): {normalized_points[i]}")
        except Exception as e:
            raise RuntimeError(f"Error processing the image: {e}")