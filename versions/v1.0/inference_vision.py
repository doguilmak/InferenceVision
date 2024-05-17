from ultralytics import YOLO
import rasterio.warp
from rasterio import features
import numpy as np
import pandas as pd

class InferenceVision:

    VERSION = "1.0"

    def __init__(self, tif_path, model_path, image_width, image_height):
        self.tif_path = tif_path
        self.model_path = model_path
        self.image_width = image_width
        self.image_height = image_height
        self.extracted_polygons_lonlat = []
        self.results = []
        self.center_coords = []
        self.extracted_polygons = []

    def normalize_center_points(self, center_points):
        normalized_points = []
        for x_center, y_center in center_points:
            normalized_x = x_center / self.image_width
            normalized_y = y_center / self.image_height
            normalized_points.append([normalized_x, normalized_y])
        return normalized_points

    def calculate_bbox_center(self, coordinates):
        x_min, y_min, x_max, y_max = coordinates
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + (width / 2)
        y_center = y_min + (height / 2)
        return x_center, y_center

    def process_image(self, build_csv=False, csv_filename=None):
        model = YOLO(self.model_path)
        # results = model.predict(self.tif_path, save=True)
        results = model.predict(self.tif_path)

        with rasterio.open(self.tif_path) as dataset:
            mask = dataset.dataset_mask()
            for geom, val in rasterio.features.shapes(mask, transform=dataset.transform):
                geom = rasterio.warp.transform_geom(dataset.crs, 'EPSG:4326', geom, precision=6)
                self.extracted_polygons.append(geom['coordinates'][0])

            self.extracted_polygons[0] = self.extracted_polygons[0][:-1]
            self.extracted_polygons_lonlat = self.extracted_polygons[0]

            for result in results:
                boxes = result.boxes
                masks = result.masks
                keypoints = result.keypoints
                probs = result.probs

            result = results[0]
            detected_objects= len(result.boxes)

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
                print("-"*30)
                class_id = result.names[box.cls[0].item()]
                print(f"Point {i}")
                print(f"Latitude: {formatted_geographic_coords[i][1]} - Longitude: {formatted_geographic_coords[i][0]} ({formatted_geographic_coords[i][1]}, {formatted_geographic_coords[i][0]})")
                print("Object type:", class_id)
                print("Object type:", box.cls[0].item())
                print("Coordinates:", box.xyxy[0].tolist())
                print("Probability:", box.conf[0].item())
                print("Bounding Box Center:", [x_center, y_center])
                print("Normalized Bounding Box Center:", [normalized_points[i][0], normalized_points[i][1]])
