import json

import numpy as np
import pandas as pd
import rasterio.warp
from rasterio import features
from ultralytics import YOLO


class InferenceVision:

    VERSION = "1.3"

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
            The number of decimal places to use for geographic coordinates.
            Default is 9.

        Raises
        ------
        ValueError
            If the TIFF file cannot be opened.

        Example
        -------
        >>> iv = InferenceVision("image.tif", "model.pt", coord_precision=6)
        """
        self.tif_path = tif_path
        self.model_path = model_path
        self.coord_precision = coord_precision

        # State — reset at each process_image() call
        self.extracted_polygons = []
        self.extracted_polygons_lonlat = []
        self.center_coords = []
        self.results = []

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
            A list of (x, y) center coordinates in pixels.

        Returns
        -------
        list of list
            A list of [norm_x, norm_y] values in the [0, 1] range.

        Example
        -------
        >>> iv = InferenceVision("image.tif", "model.pt")
        >>> iv.normalize_center_points([(100, 200), (150, 250)])
        """
        return [
            [x / self.image_width, y / self.image_height]
            for x, y in center_points
        ]

    def process_image(self, output_format=None, csv_filename=None, geojson_filename=None):
        """
        Run YOLO inference on the TIFF image and output geographic results.

        Parameters
        ----------
        output_format : str or None, optional
            Output mode. Accepted values:
            - None       : print results to console (default)
            - "csv"      : save results to a CSV file (requires csv_filename)
            - "geojson"  : save results as a GeoJSON FeatureCollection
                           (requires geojson_filename)
        csv_filename : str, optional
            Destination path for the CSV file. Required when
            output_format="csv".
        geojson_filename : str, optional
            Destination path for the GeoJSON file. Required when
            output_format="geojson".

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If output_format is "csv" but csv_filename is not provided, or
            output_format is "geojson" but geojson_filename is not provided.
        RuntimeError
            If model inference or image processing fails.

        Example
        -------
        >>> iv = InferenceVision("image.tif", "model.pt")
        >>> iv.process_image(output_format="csv", csv_filename="out.csv")
        >>> iv.process_image(output_format="geojson", geojson_filename="out.geojson")
        >>> iv.process_image()  # print to console
        """
        # --- Input validation ---
        if output_format == "csv" and not csv_filename:
            raise ValueError(
                "csv_filename must be provided when output_format='csv'."
            )
        if output_format == "geojson" and not geojson_filename:
            raise ValueError(
                "geojson_filename must be provided when output_format='geojson'."
            )

        # --- Reset state on every call to prevent accumulation ---
        self.extracted_polygons = []
        self.extracted_polygons_lonlat = []
        self.center_coords = []
        self.results = []

        # --- Model inference ---
        try:
            model = YOLO(self.model_path)
            yolo_results = model.predict(self.tif_path)
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {e}")

        try:
            with rasterio.open(self.tif_path) as dataset:
                mask = dataset.dataset_mask()
                for geom, _ in rasterio.features.shapes(mask, transform=dataset.transform):
                    geom = rasterio.warp.transform_geom(
                        dataset.crs, "EPSG:4326", geom,
                        precision=self.coord_precision
                    )
                    self.extracted_polygons.append(geom["coordinates"][0])

                self.extracted_polygons[0] = self.extracted_polygons[0][:-1]
                self.extracted_polygons_lonlat = self.extracted_polygons[0]

                result = yolo_results[0]
                self.results = result
                detected_objects = len(result.boxes)

                if detected_objects == 0:
                    print("No objects detected in the image.")
                    return

                # polygon[0][0][0] → longitude of top-left corner
                # polygon[0][0][1] → latitude  of top-left corner
                # polygon[0][2][0] → longitude of bottom-right corner
                # polygon[0][2][1] → latitude  of bottom-right corner
                lon_tl = self.extracted_polygons[0][0][0]
                lat_tl = self.extracted_polygons[0][0][1]
                lon_br = self.extracted_polygons[0][2][0]
                lat_br = self.extracted_polygons[0][2][1]

                data_list = []
                for i in range(detected_objects):
                    try:
                        box = result.boxes[i]

                        cx, cy, bw, bh = box.xywh[0].tolist()

                        norm_x = cx / self.image_width   # x-axis → longitude
                        norm_y = cy / self.image_height  # y-axis → latitude

                        self.center_coords.append([cx, cy])

                        data_list.append({
                            "index":   i,
                            "cx":      cx,
                            "cy":      cy,
                            "bw":      bw,
                            "bh":      bh,
                            "norm_x":  norm_x,
                            "norm_y":  norm_y,
                            "class_id": result.names[box.cls[0].item()],
                            "conf":    box.conf[0].item(),
                            "bbox":    box.xyxy[0].tolist(),
                        })
                    except Exception as e:
                        print(f"Warning: skipping box {i} due to error: {e}")
                        continue

                if not data_list:
                    print("All boxes were skipped due to errors.")
                    return

                # norm_arr columns: 0 = norm_x (→ lon), 1 = norm_y (→ lat)
                norm_arr = np.array([[d["norm_x"], d["norm_y"]] for d in data_list])
                lons = lon_tl + (lon_br - lon_tl) * norm_arr[:, 0]
                lats = lat_tl + (lat_br - lat_tl) * norm_arr[:, 1]

                # --- Output ---
                if output_format == "csv":
                    self._save_csv(data_list, lons, lats, csv_filename)

                elif output_format == "geojson":
                    self._save_geojson(data_list, lons, lats, geojson_filename)

                else:
                    self._print_results(data_list, lons, lats)

        except Exception as e:
            raise RuntimeError(f"Error processing the image: {e}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_csv(self, data_list, lons, lats, csv_filename):
        """Build a DataFrame from detection results and write it to CSV."""
        try:
            rows = [
                {
                    "Image":                          self.tif_path,
                    "Point":                          d["index"],
                    "Latitude":                       f"{lats[i]:.{self.coord_precision}f}",
                    "Longitude":                      f"{lons[i]:.{self.coord_precision}f}",
                    "Object Type":                    d["class_id"],
                    "Coordinates":                    d["bbox"],
                    "Confidence Score":               d["conf"],
                    "Bounding Box Center":            [d["cx"], d["cy"]],
                    "Normalized Bounding Box Center": [d["norm_x"], d["norm_y"]],
                }
                for i, d in enumerate(data_list)
            ]
            pd.DataFrame(rows).to_csv(csv_filename, index=False)
            print(f"\nCSV saved as {csv_filename}")
        except Exception as e:
            raise RuntimeError(f"Error saving CSV file: {e}")

    def _save_geojson(self, data_list, lons, lats, geojson_filename):
        """
        Build a GeoJSON FeatureCollection and write it to disk.

        Each detected object becomes a GeoJSON Point feature. The file is
        compatible with QGIS, Leaflet, Google Earth, and any OGR-based tool
        without any additional conversion.
        """
        try:
            features_list = [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            round(float(lons[i]), self.coord_precision),
                            round(float(lats[i]), self.coord_precision),
                        ],
                    },
                    "properties": {
                        "point":            d["index"],
                        "image":            self.tif_path,
                        "class":            d["class_id"],
                        "confidence":       round(d["conf"], 4),
                        "bbox_xyxy":        d["bbox"],
                        "bbox_center_px":   [round(d["cx"], 2), round(d["cy"], 2)],
                        "normalized_center":[round(d["norm_x"], 8), round(d["norm_y"], 8)],
                    },
                }
                for i, d in enumerate(data_list)
            ]

            geojson_data = {
                "type": "FeatureCollection",
                "features": features_list,
            }

            with open(geojson_filename, "w", encoding="utf-8") as f:
                json.dump(geojson_data, f, indent=2, ensure_ascii=False)

            print(f"\nGeoJSON saved as {geojson_filename}")
        except Exception as e:
            raise RuntimeError(f"Error saving GeoJSON file: {e}")

    def _print_results(self, data_list, lons, lats):
        """Print detection results to the console."""
        for i, d in enumerate(data_list):
            print(f"\nPoint {d['index']} {'-' * 20}")
            print(f"Latitude:  {lats[i]:.{self.coord_precision}f} | "
                  f"Longitude: {lons[i]:.{self.coord_precision}f}")
            print(f"Object Type: {d['class_id']}")
            print(f"Coordinates (Bounding Box): {d['bbox']}")
            print(f"Confidence Score: {d['conf']:.4f}")
            print(f"Bounding Box Center (X, Y): ({d['cx']:.2f}, {d['cy']:.2f})")
            print(f"Normalized Bounding Box Center (X, Y): "
                  f"({d['norm_x']:.6f}, {d['norm_y']:.6f})")
