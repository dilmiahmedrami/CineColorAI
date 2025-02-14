import os
import logging
import time  # Add this import
from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image, UnidentifiedImageError
import numpy as np
from sklearn.cluster import KMeans
import cv2
from collections import Counter
from webcolors import rgb_to_name
from sklearn.metrics import silhouette_score
from scipy.spatial import KDTree
import mimetypes
from werkzeug.utils import secure_filename
from io import BytesIO
from collections import defaultdict
import statistics
import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.models import ResNet50_Weights  # New import
import colorsys
from scipy.ndimage import gaussian_filter
from sklearn.metrics import pairwise_distances_argmin_min
from skimage import exposure
from skimage.filters import threshold_otsu
import base64  # Import base64 for encoding images in responses
import matplotlib.pyplot as plt  # New import for histogram plotting


# Configuration
UPLOAD_FOLDER = 'uploads'
LUT_FOLDER = 'luts'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_COLORS_DEFAULT = 20
IMAGE_RESIZE_SIZE = 256
KMEANS_INIT = 'k-means++'
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10
SILHOUETTE_SCORE_MIN_CLUSTERS = 2
WCSS_ELBOW_START_CLUSTER = 3
SILHOUETTE_K_START_CLUSTER = 2
SIMILARITY_THRESHOLD_CINEMATIC_PALETTE = 50
DEFAULT_LUT_SIZE = 33
CUSTOM_LUT_FILENAME = "custom_lut.cube"
LUTS_SUBDIR = "luts"
LUT_TITLE = "CustomLUT"


app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LUT_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Predefined list of color names and their RGB values
COLOR_NAMES = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'lime': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
    'silver': (192, 192, 192),
    'gray': (128, 128, 128),
    'maroon': (128, 0, 0),
    'olive': (128, 128, 0),
    'green': (0, 128, 0),
    'purple': (128, 0, 128),
    'teal': (0, 128, 128),
    'navy': (0, 0, 128)
}

COLOR_TREE = KDTree(list(COLOR_NAMES.values()))

def closest_color(requested_color: tuple[int, int, int]) -> str:
    """Find the closest color name from predefined list using KDTree."""
    _, index = COLOR_TREE.query(requested_color)
    return list(COLOR_NAMES.keys())[index]

def get_color_name(rgb_color: tuple[int, int, int]) -> str:
    """Get the color name from webcolors or fallback to closest color."""
    try:
        return rgb_to_name(rgb_color)
    except ValueError:
        return closest_color(rgb_color)

def determine_optimal_clusters(img_array: np.ndarray, max_clusters: int = 10) -> int:
    """Determine optimal number of clusters using WCSS and silhouette score."""
    wcss = []
    silhouettes = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=KMEANS_N_INIT)
        kmeans.fit(img_array)
        wcss.append(kmeans.inertia_)

        if k >= SILHOUETTE_SCORE_MIN_CLUSTERS:
            score = silhouette_score(img_array, kmeans.labels_)
            silhouettes.append(score)

    # Improved elbow detection using knee point detection
    deltas = np.diff(wcss)
    deltas2 = np.diff(deltas)
    optimal_k = np.argmax(deltas2) + WCSS_ELBOW_START_CLUSTER if len(deltas2) > 0 else WCSS_ELBOW_START_CLUSTER

    # Validate with silhouette scores
    if len(silhouettes) > 0:
        silhouette_k = np.argmax(silhouettes) + SILHOUETTE_K_START_CLUSTER
        optimal_k = max(optimal_k, silhouette_k)

    return min(optimal_k, max_clusters)

def rgb_to_kelvin(r: int, g: int, b: int) -> float:
    """Convert RGB to Kelvin temperature (simplified placeholder)."""
    return (r + g + b) / 3 * 100  # Simplified placeholder calculation

def rule_of_thirds_analysis(image: np.ndarray) -> dict:
    """Enhanced rule of thirds analysis with detailed metrics."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Adaptive thresholding using Otsu's method
        thresh_val = threshold_otsu(gray)
        edges = cv2.Canny(gray, 0.5 * thresh_val, thresh_val)

        height, width, _ = image.shape

        # Define rule of thirds grid lines
        grid_lines = [
            (width / 3, 0), (width * 2 / 3, 0),  # Vertical lines
            (0, height / 3), (0, height * 2 / 3),  # Horizontal lines
        ]

        # Intersection points (more precise calculation)
        intersection_points = [
            (int(width / 3), int(height / 3)),
            (int(width / 3), int(height * 2 / 3)),
            (int(width * 2 / 3), int(height / 3)),
            (int(width * 2 / 3), int(height * 2 / 3)),
        ]

        interest_values = []
        for x, y in intersection_points:
            region_size = min(50, width // 5, height // 5)  # adaptive region size
            x1 = max(0, x - region_size // 2)
            y1 = max(0, y - region_size // 2)
            x2 = min(width, x + region_size // 2)
            y2 = min(height, y + region_size // 2)
            region = image[y1:y2, x1:x2]

            if region.size > 0:  # check if region is not empty
                # Combined Interest Metric:
                # 1. Average Saturation:
                hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
                saturation = np.mean(hsv_region[:, :, 1])

                # 2. Local Contrast (Standard Deviation of Lightness):
                gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
                contrast = np.std(gray_region)

                # 3. Edge Density (Number of edges in the region):
                edges_region = cv2.Canny(gray_region, 50, 150)  # Basic edge detection
                edge_density = np.sum(edges_region) / region.size if region.size > 0 else 0  # edges per pixel

                interest = saturation * (contrast / 50) * (1 + edge_density * 10)  # Combine metrics (adjust weights)

                interest_values.append({
                    "x": x,
                    "y": y,
                    "saturation": float(saturation),
                    "contrast": float(contrast),
                    "edge_density": float(edge_density),
                    "interest": float(interest)
                })
            else:
                interest_values.append({
                    "x": x,
                    "y": y,
                    "saturation": 0.0,
                    "contrast": 0.0,
                    "edge_density": 0.0,
                    "interest": 0.0
                })

        composition_score = sum([iv["interest"] for iv in interest_values])

        return {
            "grid_lines": grid_lines,
            "intersection_points": intersection_points,
            "interest_values": interest_values,
            "composition_score": composition_score,
        }
    except Exception as e:
        logger.error(f"Error in rule_of_thirds_analysis: {e}", exc_info=True)
        return {
            "grid_lines": [],
            "intersection_points": [],
            "interest_values": [],
            "composition_score": 0.0,
        }

def extract_dominant_colors(img_array: np.ndarray, max_colors: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract dominant colors using KMeans clustering."""
    pixels = img_array.reshape(-1, 3)
    max_colors = min(max_colors, len(np.unique(pixels, axis=0))) #limit max color to amount of unique colors
    kmeans = KMeans(n_clusters=max_colors, random_state=KMEANS_RANDOM_STATE, init=KMEANS_INIT, n_init=KMEANS_N_INIT).fit(pixels)
    counts = Counter(kmeans.labels_)
    total = len(kmeans.labels_)
    percentages = np.array([count / total * 100 for count in counts.values()]) #percent of each color
    return kmeans.cluster_centers_.astype(int), percentages

def calculate_color_metrics(img_array: np.ndarray) -> dict:
    """Calculate various color metrics from the image array."""
    gray_array = np.mean(img_array, axis=2)
    hsv_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    brightness = float(gray_array.mean())
    contrast = float(gray_array.std())
    saturation = float(hsv_array[:, :, 1].mean())
    lift = img_array.min(axis=(0, 1)).astype(int).tolist()
    gamma = np.median(img_array, axis=(0, 1)).astype(int).tolist()
    gain = img_array.max(axis=(0, 1)).astype(int).tolist()
    offset = img_array.mean(axis=(0, 1)).astype(int).tolist()
    pivot = float(np.median(gray_array))
    r, g, b = img_array.mean(axis=(0, 1))
    color_temperature = float(rgb_to_kelvin(r, g, b))
    tint = float((r - g) / (r + g + 1e-5) * 100)
    exposure = brightness

    return {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "lift": lift,
        "gamma": gamma,
        "gain": gain,
        "offset": offset,
        "pivot": pivot,
        "color_temperature": color_temperature,
        "tint": tint,
        "exposure": exposure,
    }

def detect_cinematic_palettes(dominant_colors: np.ndarray) -> list[str]:
    """Detect cinematic color palettes based on dominant colors."""
    cinematic_palettes = {
        "Teal-Orange": [(0, 128, 128), (255, 165, 0)],
        "Sepia": [(112, 66, 20), (255, 228, 181)],
        "Noir": [(0, 0, 0), (255, 255, 255)]
    }
    detected_palettes = []
    for palette_name, palette_colors in cinematic_palettes.items():
        palette_tree = KDTree(palette_colors)
        distances, _ = palette_tree.query(dominant_colors)
        if np.mean(distances) < SIMILARITY_THRESHOLD_CINEMATIC_PALETTE:
            detected_palettes.append(palette_name)
    return detected_palettes

def analyze_dynamic_range(gray_array: np.ndarray) -> dict:
    """Analyze dynamic range of the image based on histogram."""
    histogram, _ = np.histogram(gray_array, bins=256, range=(0, 255))
    shadows = int(np.sum(histogram[:85]))
    midtones = int(np.sum(histogram[85:170]))
    highlights = int(np.sum(histogram[170:]))
    return {
        "shadows": shadows,
        "midtones": midtones,
        "highlights": highlights
    }

def classify_mood(saturation: float) -> str:
    """Classify mood based on saturation level."""
    if saturation < 50:
        return "desaturated"
    elif saturation > 150:
        return "oversaturated"
    return "neutral"

def detect_film_grain(gray_image: np.ndarray) -> str:
    """Detect film grain based on Laplacian variance."""
    laplacian_var = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())
    return "high" if laplacian_var > 1000 else "low"

def convert_to_serializable(obj):
    """Convert NumPy int64, float64, and NaN to Python int, float, and valid JSON values."""
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.float64):
        return float(np.nan_to_num(obj))  # Convert NaN to zero
    if isinstance(obj, np.ndarray):
        return np.nan_to_num(obj).tolist()  # Convert NaN to zero in arrays
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

def analyze_image_colors(image_path, max_colors: int = MAX_COLORS_DEFAULT):
    """Analyze image colors and calculate various color grading metrics."""
    try:
        logger.debug(f"Opening image from file-like object of type: {type(image_path)}")
        with Image.open(image_path) as image:
            logger.debug(f"Image opened successfully. Format: {image.format}, Size: {image.size}")
            image = image.convert('RGB')
            image = image.resize((IMAGE_RESIZE_SIZE, IMAGE_RESIZE_SIZE), resample=Image.LANCZOS)
            logger.debug(f"Image converted to RGB and resized to {IMAGE_RESIZE_SIZE}x{IMAGE_RESIZE_SIZE}")
            img_array = np.array(image)
            logger.debug(f"Converted image to array with shape: {img_array.shape}")

            # Extract dominant colors and percentages
            dominant_colors, color_percentages_array = extract_dominant_colors(img_array, max_colors)
            color_percentages = {int(i): float(color_percentages_array[i]) for i in range(len(color_percentages_array))} # Convert back to dict if needed for JSON

            logger.debug(f"K-Means clustering complete. Dominant colors: {dominant_colors}")
            logger.debug(f"Color percentages calculated: {color_percentages}")

            color_names = [get_color_name(tuple(color)) for color in dominant_colors]
            logger.debug(f"Color names determined: {color_names}")

            # Calculate average color
            avg_color = img_array.reshape(-1, 3).mean(axis=0).astype(int)
            logger.debug(f"Average color calculated: {avg_color}")

            # Single grayscale conversion to be reused
            gray_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            logger.debug("Image converted to grayscale.")

            # Calculate color metrics using gray_array
            brightness = float(gray_array.mean())
            contrast = float(gray_array.std())

            # Calculate color metrics
            color_metrics = calculate_color_metrics(img_array)
            logger.debug(f"Color metrics calculated: {color_metrics}")

            # Detect cinematic palettes
            detected_palettes = detect_cinematic_palettes(dominant_colors)
            logger.debug(f"Detected cinematic palettes: {detected_palettes}")

            # Analyze histogram for dynamic range
            dynamic_range = analyze_dynamic_range(gray_array)
            logger.debug(f"Dynamic range analysis: {dynamic_range}")

            # Classify saturation and mood
            mood = classify_mood(color_metrics['saturation'])
            logger.debug(f"Mood classification: {mood}")

            # Film grain and texture detection (optional)
            film_grain = detect_film_grain(gray_array)
            logger.debug(f"Film grain detection: {film_grain}")

            # Convert RGB to LAB for more accurate color extraction
            lab_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            lab_dominant_colors = cv2.cvtColor(dominant_colors.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).tolist()
            logger.debug(f"LAB dominant colors: {lab_dominant_colors}")

            # Perform rule of thirds analysis
            rule_of_thirds_result = rule_of_thirds_analysis(img_array)
            logger.debug(f"Rule of Thirds analysis: {rule_of_thirds_result}")

            # Calculate sharpness
            sharpness = cv2.Laplacian(gray_array, cv2.CV_64F).var()
            logger.debug(f"Sharpness calculated: {sharpness}")

            # Calculate color harmony
            color_harmony = calculate_color_harmony(dominant_colors, color_percentages)
            logger.debug(f"Color harmony calculated: {color_harmony}")

            # Divide the image into regions for LUT generation
            regions = []
            region_size = 64  # Example region size
            height, width, _ = img_array.shape
            logger.debug(f"Dividing image into regions with region size {region_size}, image dimensions: {width}x{height}")
            for y in range(0, height, region_size):
                for x in range(0, width, region_size):
                    region = img_array[y:y+region_size, x:x+region_size]
                    if region.size == 0:
                        continue
                    avg_color = region.mean(axis=(0, 1)).astype(int)
                    regions.append({
                        "x": x,
                        "y": y,
                        "width": int(region.shape[1]),
                        "height": int(region.shape[0]),
                        "average_color_rgb": avg_color.tolist()
                    })
            # Group regions into a grid for interactive LUT
            cols = max(1, width // region_size)
            interactive_lut = [regions[i:i+cols] for i in range(0, len(regions), cols)]

            # Generate LUT file
            lut_filename = generate_lut_file_interactive(interactive_lut)
            logger.debug(f"LUT file generated: {lut_filename}")

            dominant_color = dominant_colors[0]

            dominant_hue = rgb_to_hsv(*dominant_color)[0]
            secondary_hue = (dominant_hue + 60) % 360
            accent_hue = (dominant_hue + 180) % 360

            secondary_color = hsv_to_rgb(secondary_hue, 1, 1)
            accent_color = hsv_to_rgb(accent_hue, 1, 1)

            graded_image = apply_60_30_10(img_array.copy(), dominant_color, secondary_color, accent_color)
            if (graded_image is None):
                graded_image = img_array  # Use original image if grading fails

            resnet_features = extract_resnet_features(image)

            # Perform three-way color balance adjustment (example values - replace with user input later)
            shadows_balance = [0, 0, 0]
            midtones_balance = [0, 0, 0]
            highlights_balance = [0, 0, 0]
            balanced_image = apply_three_way_color_balance(img_array.copy(), shadows_balance, midtones_balance, highlights_balance)

            # Calculate histogram for visualization
            histogram, _ = np.histogram(gray_array, bins=256, range=(0, 255))

            # Generate LUT data
            lut_size = DEFAULT_LUT_SIZE
            lut_data = np.zeros((lut_size, lut_size, lut_size, 4), dtype=np.uint8)
            
            # Create identity LUT modified by dominant colors
            for b in range(lut_size):
                for g in range(lut_size):
                    for r in range(lut_size):
                        # Base values
                        base_r = int((r / (lut_size - 1)) * 255)
                        base_g = int((g / (lut_size - 1)) * 255)
                        base_b = int((b / (lut_size - 1)) * 255)
                        
                        # Apply color influence from dominant colors
                        if len(dominant_colors) > 0:
                            influence = 0.2  # Strength of color influence
                            dc = dominant_colors[0]  # Use primary dominant color
                            
                            # Blend with dominant color
                            final_r = int(base_r * (1 - influence) + dc[0] * influence)
                            final_g = int(base_g * (1 - influence) + dc[1] * influence)
                            final_b = int(base_b * (1 - influence) + dc[2] * influence)
                            
                            lut_data[r, g, b] = [final_r, final_g, final_b, 255]
                        else:
                            lut_data[r, g, b] = [base_r, base_g, base_b, 255]

            # Flatten the LUT data for transfer
            flat_lut = lut_data.reshape(-1).tolist()
            
            logger.debug(f"Generated LUT data with shape: {lut_data.shape}")
            logger.debug(f"First few LUT values: {flat_lut[:12]}")  # Log first few values for debugging

            # Calculate comprehensive metrics
            comprehensive_metrics = calculate_comprehensive_metrics(img_array)
            logger.debug(f"Comprehensive metrics calculated: {comprehensive_metrics}")

            # Calculate RGB histogram
            rgb_histogram = calculate_rgb_histogram(img_array)
            logger.debug(f"RGB histogram calculated: {rgb_histogram}")

        result = {
            "dominant_colors": convert_to_serializable(dominant_colors),
            "color_percentages": convert_to_serializable(color_percentages),
            "color_names": convert_to_serializable(color_names),
            "average_color": convert_to_serializable(avg_color),
            **convert_to_serializable(color_metrics), # Include brightness, contrast, etc from color_metrics dict
            "cinematic_palettes": convert_to_serializable(detected_palettes),
            "dynamic_range": convert_to_serializable(dynamic_range),
            "mood": convert_to_serializable(mood),
            "film_grain": convert_to_serializable(film_grain),
            "lab_dominant_colors": convert_to_serializable(lab_dominant_colors),
            "rule_of_thirds": convert_to_serializable(rule_of_thirds_result),  # Add rule of thirds result
            "sharpness": convert_to_serializable(sharpness),  # Add sharpness metric
            "color_harmony": convert_to_serializable(color_harmony),  # Add color harmony metric
            "regions": convert_to_serializable(regions),  # Add regions for LUT generation
            "interactive_lut": convert_to_serializable(interactive_lut),  # New key for interactive LUT grid
            "histogram": convert_to_serializable(histogram.tolist()),  # Added histogram data for visualization
            "lut_filename": convert_to_serializable(lut_filename),  # Add LUT filename to result
            "graded_image_base64": convert_to_serializable(image_to_base64(graded_image)), # Return base64 encoded image
            "resnet_features": convert_to_serializable(resnet_features.tolist()),
            "balanced_image_base64": convert_to_serializable(image_to_base64(balanced_image)), # Return base64 encoded image
            "lut_data": convert_to_serializable(flat_lut),  # Add LUT data to result
            "comprehensive_metrics": convert_to_serializable(comprehensive_metrics),
            "rgb_histogram": convert_to_serializable(rgb_histogram),
        }
        logger.debug(f"Final analysis result keys: {list(result.keys())}")

        # Generate enhanced LUT
        lut_filename_enhanced = generate_lut_from_analysis(result)
        if lut_filename_enhanced:
            result["enhanced_lut_filename"] = lut_filename_enhanced # Use a different key to avoid overwriting

        return result
    except UnidentifiedImageError as e:
        logger.error("Unidentified image error.", exc_info=True)
        raise
    except FileNotFoundError as e:
        logger.error("Image file not found.", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise

def image_to_base64(image_np: np.ndarray) -> str:
    """Encode a numpy image array to base64 string."""
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


def generate_lut_file_interactive(interactive_lut: list[list[dict]]) -> str:
    """Generate and save the LUT file using adjusted colors from interactive regions."""
    lut_dir = LUT_FOLDER
    os.makedirs(lut_dir, exist_ok=True)
    lut_filename = os.path.join(lut_dir, "lut.cube")
    lut_size = DEFAULT_LUT_SIZE
    try:
        with open(lut_filename, "w") as lut_file:
            lut_file.write("TITLE \"Interactive LUT\"\n")
            lut_file.write(f"LUT_3D_SIZE {lut_size}\n")
            # Write a simple 3D LUT grid as a placeholder
            for b in range(lut_size):
                for g in range(lut_size):
                    for r in range(lut_size):
                        line = f"{(r/(lut_size-1)):.6f} {(g/(lut_size-1)):.6f} {(b/(lut_size-1)):.6f}"
                        lut_file.write(line + "\n")
        return "lut.cube"
    except Exception as e:
        logger.error(f"Error generating interactive LUT file: {e}", exc_info=True)
        return None

def generate_lut_from_analysis(analysis_result: dict, lut_size: int = DEFAULT_LUT_SIZE) -> str:
    """Simplified 3D LUT generator."""
    dominant_color = np.array(analysis_result['dominant_colors'][0]) / 255.0

    lut_data = np.zeros((lut_size, lut_size, lut_size, 3))

    for b in range(lut_size):
        for g in range(lut_size):
            for r in range(lut_size):
                # Normalize grid values to 0-1 range
                rgb = np.array([r, g, b]) / (lut_size - 1)

                # Apply a simple color shift based on the dominant color
                influence = 0.1  # Strength of color influence
                rgb = rgb * (1 - influence) + dominant_color * influence

                lut_data[r, g, b] = rgb * 255

    return generate_lut_file_from_array(lut_data.astype(np.uint8), filename=CUSTOM_LUT_FILENAME, lut_size=lut_size)

def generate_lut_file_from_array(lut_data: np.ndarray, filename: str = CUSTOM_LUT_FILENAME, lut_size: int = DEFAULT_LUT_SIZE) -> str:
    """Generates and saves a .cube LUT file from a NumPy array, handling 3 or 4 channels, with improved error handling."""
    lut_dir = LUT_FOLDER
    os.makedirs(lut_dir, exist_ok=True)
    lut_filename = os.path.join(lut_dir, filename)

    try:
        # Input validation: Check if lut_data is empty or not a NumPy array
        if not isinstance(lut_data, np.ndarray) or lut_data.size == 0:
            logger.error(f"Invalid LUT data: lut_data is empty or not a NumPy array.")
            return None

        # Ensure lut_data is 3D or 4D and has the correct shape
        if lut_data.ndim not in (3, 4) or lut_data.shape[:3] != (lut_size, lut_size, lut_size):
            logger.error(f"Invalid LUT data shape: {lut_data.shape}. Expected ({lut_size}, {lut_size}, {lut_size}, [3 or 4])")
            return None

        # Check for 3 or 4 channels
        num_channels = lut_data.shape[3] if lut_data.ndim == 4 else 3
        if num_channels not in (3, 4):
            logger.error(f"Invalid number of channels: Expected 3 or 4, got {num_channels}")
            return None

        with open(lut_filename, "w") as lut_file:
            lut_file.write(f"TITLE \"{LUT_TITLE}\"\n")
            lut_file.write(f"LUT_3D_SIZE {lut_size}\n")
            lut_file.write("DOMAIN_MIN 0.0 0.0 0.0\n")  # Add DOMAIN_MIN
            lut_file.write("DOMAIN_MAX 1.0 1.0 1.0\n")  # Add DOMAIN_MAX

            for z in range(lut_size):
                for y in range(lut_size):
                    for x in range(lut_size):
                        if num_channels == 4:
                            r, g, b, _ = lut_data[x, y, z]  # Ignore alpha
                        else:
                            r, g, b = lut_data[x, y, z]
                        # Ensure values are within 0-255 range and convert to 0.0-1.0 range
                        r = max(0, min(255, int(r)))
                        g = max(0, min(255, int(g)))
                        b = max(0, min(255, int(b)))
                        lut_file.write(f"{(r / 255.0):.6f} {(g / 255.0):.6f} {(b / 255.0):.6f}\n")

        logger.info(f"LUT file created successfully: {lut_filename}")  # Log successful creation
        return filename
    except Exception as e:
        logger.error(f"Error saving LUT file: {e}", exc_info=True)
        return None

@app.route('/download_lut/<filename>')
def download_lut(filename: str):
    lut_type = request.args.get('lut_type', 'cube').lower()
    allowed_types = ['cube', '3dl', 'lut']
    if lut_type not in allowed_types:
        lut_type = 'cube'
    filename = f"custom_lut.{lut_type}"
    lut_path = os.path.join(LUT_FOLDER, filename)
    if os.path.exists(lut_path):
        return send_file(lut_path, as_attachment=True)
    else:
        return jsonify({"error": "LUT file not found."}), 404

@app.route('/analyze', methods=['POST'])
def analyze_image_route():
    """API endpoint to analyze a reference image."""
    logger.info(f"Request from IP: {request.remote_addr}")
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        logger.debug(f"Uploaded file extension not allowed: {file.filename}")
        return jsonify({"error": "Invalid file type. Only PNG, JPG, and JPEG are supported."}), 400

    try:
        file_content = file.read()
        logger.debug(f"Read file of length: {len(file_content)} bytes, mimetype: {file.mimetype}")
        file_stream = BytesIO(file_content)
        file_stream.seek(0)

        analysis_result = analyze_image_colors(file_stream)

        if analysis_result:
            logger.debug("Analysis produced results successfully.")
            return jsonify(analysis_result), 200
        else:
            logger.error("Analysis result empty after processing.")
            return jsonify({"error": "Image analysis failed."}), 500

    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image file."}), 400
    except Exception as e:
        logger.error("Error analyzing image:", exc_info=True)
        return jsonify({"error": f"Error analyzing image: {str(e)}"}), 500

@app.route('/')
def index():
    return render_template('main.html')

# New documentation route
@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

# New helper function for image optimization
def convert_image_to_format(image: Image.Image, fmt: str = "WEBP", quality: int = 80) -> BytesIO:
    out = BytesIO()
    image.save(out, format=fmt, quality=quality)
    out.seek(0)
    return out

@app.route('/generate_lut', methods=['POST'])
def generate_lut_route():
    """API endpoint to generate and apply LUT to a target image."""
    target_file = request.files.get('target_file')
    if not target_file:
        logger.debug("No target image provided for LUT generation.")
        return jsonify({"error": "No target image provided."}), 400

    if not allowed_file(target_file.filename):
        logger.debug(f"Invalid target image mimetype for LUT generation: {target_file.mimetype}")
        return jsonify({"error": "Invalid target image type. Only PNG, JPG, and JPEG are supported."}), 400

    try:
        logger.debug(f"Target file received for LUT generation: {target_file.filename}, mimetype: {target_file.mimetype}")
        target_image = Image.open(target_file).convert('RGB')
        logger.debug(f"Target image opened successfully for LUT generation. Size: {target_image.size}, Mode: {target_image.mode}")

        # Apply the LUT to the target image (Placeholder - replace with actual LUT application)
        lut_applied_image = apply_cube_lut(target_image) # Replace this with actual LUT application logic
        logger.debug("LUT applied to target image successfully (placeholder).")

        # Save the LUT applied image to a temporary file
        temp_file = convert_image_to_format(lut_applied_image, fmt="WEBP", quality=80)

        return send_file(temp_file, mimetype='image/webp', as_attachment=True, download_name='lut_applied_image.webp')

    except Exception as e:
        logger.error(f"Error applying cube LUT: {e}", exc_info=True)
        return jsonify({"error": "Error applying LUT."}), 500

def apply_cube_lut(image: np.ndarray, lut_data: np.ndarray) -> np.ndarray:
    """Applies a 3D LUT to an image using OpenCV."""
    # Ensure the image is in the correct format
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # OpenCV's LUT function requires a 2D LUT for 3-channel images
    if lut_data.ndim == 4:  # If it's a 4D LUT (includes alpha)
        lut_data = lut_data[:, :, :, :3]  # Use only RGB channels

    # Reshape the LUT for OpenCV
    lut_size = lut_data.shape[0]
    lut_1d = lut_data.reshape((lut_size * lut_size * lut_size, 3))

    # Apply the LUT using cv2.LUT
    graded_image = cv2.LUT(image, lut_1d)
    return graded_image

@app.route('/apply_color_balance', methods=['POST'])
def apply_color_balance_route():
    """API endpoint to apply three-way color balance."""
    try:
        data = request.json
        if not data or not all(k in data for k in ("shadows", "midtones", "highlights")):
            return jsonify({'error': 'Invalid color balance parameters.'}), 400

        # Get base64 encoded image from request
        base64_image = data.get('base64_image')
        if not base64_image:
            return jsonify({'error': 'No base64 image provided for color balance.'}), 400

        try:
            image_data = base64.b64decode(base64_image)
            image_np_array = np.frombuffer(image_data, np.uint8)
            original_image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) # Convert to RGB
        except Exception as decode_err:
            logger.error(f"Error decoding base64 image: {decode_err}", exc_info=True)
            return jsonify({'error': 'Error decoding base64 image.'}), 400

        # Apply color balance adjustments
        adjusted = apply_three_way_color_balance(
            original_image,
            data['shadows'],
            data['midtones'],
            data['highlights']
        )

        # Convert to base64 for response
        balanced_image_base64 = image_to_base64(adjusted)
        return jsonify({
            'balanced_image_base64': balanced_image_base64
        }), 200

    except Exception as e:
        logger.error(f"Error applying color balance: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/apply_advanced_adjustments', methods=['POST'])
def apply_advanced_adjustments_route():
    """API endpoint to apply advanced color adjustments (placeholder)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No adjustment data provided."}), 400

    curves = data.get('curves', {})
    hsl = data.get('hsl', {})
    selective_color = data.get('selectiveColor', {})
    split_toning = data.get('splitToning', {})
    temp_tint = data.get('tempTint', {})
    vibrance = data.get('vibrance', 0)
    base64_image = data.get('base64_image')

    if not base64_image:
        return jsonify({"error": "No base64 image provided for adjustments."}), 400

    try:
        image_data = base64.b64decode(base64_image)
        image_np_array = np.frombuffer(image_data, np.uint8)
        original_image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        adjusted_image = apply_advanced_adjustments_to_image(original_image, curves, hsl, selective_color, split_toning, temp_tint, vibrance)
        adjusted_image_str = image_to_base64(adjusted_image)
        return jsonify({"adjusted_image_base64": adjusted_image_str}), 200

    except Exception as e:
        logger.error(f"Error applying advanced adjustments: {e}", exc_info=True)
        return jsonify({"error": "Error applying advanced adjustments."}), 500

def apply_advanced_adjustments_to_image(image: np.ndarray, curves: dict, hsl: dict, selective_color: dict, split_toning: dict, temp_tint: dict, vibrance: float) -> np.ndarray:
    """Apply advanced color adjustments to an image (placeholder function)."""
    # Placeholder function - Implement actual adjustments here based on parameters
    # This is where you would integrate libraries or custom code for curves, HSL, etc.
    return image


def load_lut_data(lut_filename: str) -> np.ndarray:
    """Loads LUT data from a .cube or .3dl file."""
    lut_path = os.path.join(LUT_FOLDER, lut_filename)
    if not os.path.exists(lut_path):
        return None

    if lut_filename.lower().endswith('.cube'):
        try:
            with open(lut_path, 'r') as f:
                lines = f.readlines()

            lut_size = None
            lut_data = []
            for line in lines:
                line = line.strip()
                if line.startswith('LUT_3D_SIZE'):
                    lut_size = int(line.split()[-1])
                elif len(line.split()) == 3:  # Expecting 3 values per line
                    try:
                        r, g, b = map(float, line.split())
                        lut_data.append([int(r * 255), int(g * 255), int(b * 255)]) # Convert back to 0-255 range
                    except ValueError:
                        continue # Skip lines that don't contain valid float triples

            if lut_size is None or len(lut_data) != lut_size**3:
                logger.error(f"Invalid .cube file format or size mismatch: {lut_filename}")
                return None

            return np.array(lut_data).reshape((lut_size, lut_size, lut_size, 3))

        except Exception as e:
            logger.error(f"Error reading .cube file: {e}", exc_info=True)
            return None

    elif lut_filename.lower().endswith('.3dl'):
        try:
            with open(lut_path, 'r') as f:
                lines = f.readlines()

            lut_data = []
            for line in lines:
                line = line.strip()
                if len(line.split()) == 3:
                    try:
                        r, g, b = map(int, line.split())
                        lut_data.append([r, g, b])
                    except ValueError:
                        continue # Skip lines that don't contain valid int triples

            # Determine LUT size from the number of data points (cube root)
            lut_size = round(len(lut_data) ** (1/3))
            if len(lut_data) != lut_size**3:
                logger.error(f"Invalid .3dl file format or size mismatch: {lut_filename}")
                return None

            return np.array(lut_data).reshape((lut_size, lut_size, lut_size, 3))

        except Exception as e:
            logger.error(f"Error reading .3dl file: {e}", exc_info=True)
            return None
    else:
        logger.error(f"Unsupported LUT file format: {lut_filename}")
        return None

@app.route('/test_lut', methods=['POST'])
def test_lut():
    """Tests the application of a LUT to the test image."""
    try:
        test_image_path = os.path.join(app.static_folder, 'Test.jpg')
        logger.debug(f"Test image path: {test_image_path}")

        if not os.path.exists(test_image_path):
            logger.error("Test image not found.")
            return jsonify({"error": "Test image not found."}), 404

        test_image = cv2.imread(test_image_path)
        if test_image is None:
            logger.error("Failed to load test image.")
            return jsonify({"error": "Failed to load test image."}), 500
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        logger.debug("Test image loaded and converted to RGB.")

        lut_intensity = float(request.form.get('lut_intensity', 100)) / 100
        logger.debug(f"LUT intensity: {lut_intensity}")

        # Prioritize LUT data from analysis result
        lut_data_base64 = request.form.get('lut_data')
        if lut_data_base64:
            logger.debug("Using LUT data from analysis result.")
            lut_data_bytes = base64.b64decode(lut_data_base64)
            lut_data = np.frombuffer(lut_data_bytes, dtype=np.uint8)
            # Determine lut_size based on the length of the data
            lut_size = int(round((len(lut_data) / 4) ** (1/3))) # Assuming 4 channels (RGBA)
            if len(lut_data) != lut_size * lut_size * lut_size * 4:
                logger.error("Invalid LUT data size from analysis result.")
                return jsonify({"error": "Invalid LUT data size from analysis result."}), 400
            lut_data = lut_data.reshape((lut_size, lut_size, lut_size, 4))
            logger.debug(f"Reshaped LUT data to: {lut_data.shape}")

        else:
            lut_type = request.form.get('lut_type', 'cube').lower()
            allowed_types = ['cube', '3dl']
            if lut_type not in allowed_types:
                lut_type = 'cube'
            custom_lut_filename = f"custom_lut.{lut_type}"
            logger.debug(f"LUT type: {lut_type}, filename: {custom_lut_filename}")

            lut_data = load_lut_data(custom_lut_filename)
            logger.debug(f"LUT data loaded: {lut_data is not None}")

            if lut_data is None:
                lut_size = DEFAULT_LUT_SIZE
                lut_data = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.uint8)
                for r in range(lut_size):
                    for g in range(lut_size):
                        for b in range(lut_size):
                            lut_data[r, g, b] = [
                                int(r * 255 / (lut_size - 1)),
                                int(g * 255 / (lut_size - 1)),
                                int(b * 255 / (lut_size - 1)),
                            ]
                logger.warning("Using default identity LUT.")

        lut_applied = apply_cube_lut(test_image, lut_data)
        if lut_applied is None:
            logger.error("Failed to apply LUT.")
            return jsonify({"error": "Failed to apply LUT."}), 500
        logger.debug("LUT applied to test image.")

        blended_image = cv2.addWeighted(test_image, 1 - lut_intensity, lut_applied, lut_intensity, 0)
        logger.debug("LUT blended with original image.")

        return jsonify({
            "standard_image_base64": image_to_base64(test_image),
            "blended_image_base64": image_to_base64(blended_image),
            "lut_filename": "custom_lut.cube"  # Always return "custom_lut.cube" for consistency
        }), 200

    except Exception as e:
        logger.error(f"Error in test_lut: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def calculate_color_harmony(dominant_colors: np.ndarray, color_percentages: dict) -> dict:
    """Optimized color harmony analysis with vectorized operations."""
    hues = np.array([colorsys.rgb_to_hsv(*c)[0] * 360 for c in dominant_colors]) # Convert hue to degrees
    percentages_np = np.array(list(color_percentages.values())) # Ensure percentages are numpy array

    # Vectorized hue differences
    hue_diff = np.abs(hues[:, None] - hues)
    np.fill_diagonal(hue_diff, 0)  # Ignore self-comparisons
    hue_diff = np.minimum(hue_diff, 360 - hue_diff) # Handle hue wrapping

    # Find significant pairs (top 25% of percentage combinations)
    sig_pairs = np.argsort(percentages_np[:, None] * percentages_np, axis=None)[::-1]
    sig_pairs = [(i // len(hues), i % len(hues)) for i in sig_pairs]

    harmony_types = defaultdict(float)
    for i, j in sig_pairs[:len(sig_pairs) // 4]:
        diff = hue_diff[i, j]
        weight = percentages_np[i] * percentages_np[j]

        if 150 <= diff <= 210: # Widened ranges for more robust detection
            harmony_types['complementary'] += weight
        elif 10 <= diff <= 50: # Widened ranges
            harmony_types['analogous'] += weight
        elif 100 <= diff <= 140: # Widened ranges
            harmony_types['triadic'] += weight
        elif 60 <= diff <= 120: # Widened ranges
            harmony_types['tetradic'] += weight

    if not harmony_types:
        return {"harmony": "neutral"}

    dominant_harmony = max(harmony_types, key=harmony_types.get)
    return {
        "harmony": dominant_harmony,
        "confidence": harmony_types[dominant_harmony]
    }

# ResNet-50 setup (outside of the function)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
modules = list(resnet50.children())[:-1]
feature_extractor = torch.nn.Sequential(*modules).to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_resnet_features(image: Image.Image) -> np.ndarray:
    """Extract features from an image using ResNet50."""
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(image_tensor)
    return features.cpu().numpy().flatten()

def apply_60_30_10(image: np.ndarray, dominant_color: tuple[int, int, int], secondary_color: tuple[int, int, int], accent_color: tuple[int, int, int], tolerance: int = 30) -> np.ndarray:
    """Improved color grading using color balance adjustments."""
    # Convert to LAB color space for better perceptual adjustments
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Calculate color ratios in LAB space for better color manipulation
    dom_lab = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_RGB2LAB)[0, 0]
    sec_lab = cv2.cvtColor(np.uint8([[secondary_color]]), cv2.COLOR_RGB2LAB)[0, 0]
    acc_lab = cv2.cvtColor(np.uint8([[accent_color]]), cv2.COLOR_RGB2LAB)[0, 0]

    # Apply color balance adjustments more subtly using LAB 'a' and 'b' channels
    a_adjust = (sec_lab[1] - dom_lab[1]) * 0.3 # Reduced adjustment factor
    b_adjust = (acc_lab[2] - dom_lab[2]) * 0.3 # Reduced adjustment factor

    a_adjusted = np.clip(a + a_adjust, 0, 255).astype(np.uint8)
    b_adjusted = np.clip(b + b_adjust, 0, 255).astype(np.uint8)

    # Merge channels and convert back
    adjusted_lab = cv2.merge([l, a_adjusted, b_adjusted])
    return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2RGB)


def rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Converts RGB to HSV. Optimized version."""
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r_norm, g_norm, b_norm)
    cmin = min(r_norm, g_norm, b_norm)
    delta = cmax - cmin

    if delta == 0:
        h = 0
    elif cmax == r_norm:
        h = 60 * (((g_norm - b_norm) / delta) % 6)
    elif cmax == g_norm:
        h = 60 * (((b_norm - r_norm) / delta) + 2)
    elif cmax == b_norm:
        h = 60 * (((r_norm - g_norm) / delta) + 4)
    else: # Should not happen, but for robustness
        h = 0

    if cmax == 0:
        s = 0
    else:
        s = delta / cmax

    v = cmax
    return h, s, v


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Converts HSV to RGB."""
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = h60 - int(h60)
    p = v * (1.0 - s)
    q = v * (1.0 - (h60f * s))
    t = v * (1.0 - ((1.0 - h60f) * s))

    if h60 < 1:
        r, g, b = v, t, p
    elif h60 < 2:
        r, g, b = q, v, p
    elif h60 < 3:
        r, g, b = p, v, t
    elif h60 < 4:
        r, g, b = p, q, v
    elif h60 < 5:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return int(r * 255), int(g * 255), int(b * 255)

def apply_three_way_color_balance(image: np.ndarray, shadows: list[int], midtones: list[int], highlights: list[int]) -> np.ndarray:
    """Apply three-way color balance adjustments to an image."""
    # Convert to LAB color space for better perceptual adjustments
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply color balance adjustments
    l_adjusted = np.clip(l.astype(int) + shadows[0] + midtones[0] + highlights[0], 0, 255).astype(np.uint8)
    a_adjusted = np.clip(a.astype(int) + shadows[1] + midtones[1] + highlights[1], 0, 255).astype(np.uint8)
    b_adjusted = np.clip(b.astype(int) + shadows[2] + midtones[2] + highlights[2], 0, 255).astype(np.uint8)

    # Merge channels and convert back
    adjusted_lab = cv2.merge([l_adjusted, a_adjusted, b_adjusted])
    return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2RGB)

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/apply_tone_mapping', methods=['POST'])
def apply_tone_mapping_route():
    """API endpoint to apply tone mapping and adjustments."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No tone mapping data provided."}), 400

    operator = data.get('operator', 'reinhard')
    shadows = data.get('shadows', 50)
    highlights = data.get('highlights', 50)
    blacks = data.get('blacks', 0)
    whites = data.get('whites', 100)
    midtone_contrast = data.get('midtone_contrast', 50)
    base64_image = data.get('base64_image')

    if not base64_image:
        return jsonify({"error": "No base64 image provided for tone mapping."}), 400

    try:
        image_data = base64.b64decode(base64_image)
        image_np_array = np.frombuffer(image_data, np.uint8)
        original_image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        adjusted_image = apply_tone_mapping(original_image, operator, shadows, highlights, blacks, whites, midtone_contrast)
        adjusted_image_str = image_to_base64(adjusted_image)
        return jsonify({"adjusted_image_base64": adjusted_image_str}), 200

    except Exception as e:
        logger.error(f"Error applying tone mapping: {e}", exc_info=True)
        return jsonify({"error": "Error applying tone mapping."}), 500

def calculate_comprehensive_metrics(img_array: np.ndarray) -> dict:
    """Calculate comprehensive metrics including black/white points, contrast ratio, and clipping indicators."""
    gray_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    black_point = int(np.min(gray_array))
    white_point = int(np.max(gray_array))
    contrast_ratio = white_point / (black_point + 1e-5)  # Avoid division by zero
    clipping_indicators = {
        "black_clipping": np.sum(gray_array == 0),
        "white_clipping": np.sum(gray_array == 255)
    }
    shadow_range = np.mean(gray_array[gray_array < 85])
    midtone_range = np.mean(gray_array[(gray_array >= 85) & (gray_array < 170)])
    highlight_range = np.mean(gray_array[gray_array >= 170])

    return {
        "blackPoint": black_point,
        "whitePoint": white_point,
        "contrastRatio": contrast_ratio,
        "clippingIndicators": clipping_indicators,
        "shadowRange": shadow_range,
        "midtoneRange": midtone_range,
        "highlightRange": highlight_range
}

def calculate_rgb_histogram(img_array: np.ndarray) -> dict:
    """Calculate RGB histograms for the image."""
    red_hist = cv2.calcHist([img_array], [0], None, [256], [0, 256]).flatten()
    green_hist = cv2.calcHist([img_array], [1], None, [256], [0, 256]).flatten()
    blue_hist = cv2.calcHist([img_array], [2], None, [256], [0, 256]).flatten()
    return {
        "red": red_hist.tolist(),
        "green": green_hist.tolist(),
        "blue": blue_hist.tolist()
    }

if __name__ == '__main__':
    app.run(port=5000, debug=True)
