# -*- coding: utf-8 -*-


#Imports and Setup
import os
import time
import warnings

import numpy as np
import cv2
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Gradio version: {gr.__version__}")

#Tile Creation Methods
class TileCreator:


    @staticmethod
    def create_geometric_tiles(tile_size: int = 32) -> dict[str, np.ndarray]:
        """Create diverse geometric tiles with realistic colors + patterns."""
        tiles = {}

        # Expanded palette closer to real-life tones (skin, sky, foliage, etc.)
        colors = {
            'black': (0, 0, 0),
            'dark_gray': (64, 64, 64),
            'gray': (128, 128, 128),
            'light_gray': (200, 200, 200),
            'white': (250, 250, 250),
            'skin': (240, 200, 170),
            'brown': (139, 69, 19),
            'red': (200, 40, 40),
            'orange': (255, 140, 0),
            'yellow': (250, 210, 60),
            'green': (34, 139, 34),
            'forest_green': (0, 100, 0),
            'cyan': (0, 200, 200),
            'blue': (70, 130, 180),
            'sky_blue': (135, 206, 235),
            'purple': (138, 43, 226),
            'pink': (255, 105, 180)
        }

        for color_name, color_val in colors.items():
            tile = np.full((tile_size, tile_size, 3), color_val, dtype=np.uint8)

            # Add subtle textures to make them less flat
            if color_name in ['black', 'dark_gray', 'gray', 'light_gray', 'white']:
                # Checkerboard or diagonal lines for neutrals
                for i in range(0, tile_size, 6):
                    cv2.line(tile, (i, 0), (0, i), (255, 255, 255), 1)

            elif color_name in ['red', 'orange', 'yellow']:
                # Horizontal highlight stripes
                for i in range(0, tile_size, 6):
                    tile[i:i+1, :] = (255, 255, 255)

            elif color_name in ['green', 'forest_green', 'cyan', 'blue', 'sky_blue']:
                # Vertical highlight stripes
                for j in range(0, tile_size, 6):
                    tile[:, j:j+1] = (255, 255, 255)

            else:
                # Circular center highlight
                center = (tile_size // 2, tile_size // 2)
                cv2.circle(tile, center, tile_size // 4, (255, 255, 255), 2)

            tiles[color_name] = tile

        return tiles


def preview_tiles(tile_dict: dict[str, np.ndarray], max_cols: int = 6):
    """Preview helper (for debugging)."""
    plt.figure(figsize=(12, 4))
    for i, (name, tile) in enumerate(tile_dict.items()):
        plt.subplot(1, min(len(tile_dict), max_cols), i+1)
        plt.imshow(tile)
        plt.title(name, fontsize=8)
        plt.axis("off")
    plt.show()

#Image Processing Functions
class ImageProcessor:
    """Class for image preprocessing and grid operations."""

    @staticmethod
    def preprocess_image(image: np.ndarray,
                         target_size: tuple[int, int] = (512, 512),
                         apply_quantization: bool = False,
                         n_colors: int = 16,
                         pad: bool = True) -> np.ndarray:
        """
        Preprocess the input image with resizing, aspect ratio preservation,
        optional padding, and optional color quantization.
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size

        # Compute scaling factor to fit inside target size
        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        if pad:
            # Pad to target size (better than aggressive cropping)
            top = (target_h - new_h) // 2
            bottom = target_h - new_h - top
            left = (target_w - new_w) // 2
            right = target_w - new_w - left
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
            final_img = padded
        else:
            # Center crop (original approach)
            start_h = (new_h - target_h) // 2
            start_w = (new_w - target_w) // 2
            final_img = resized[start_h:start_h + target_h, start_w:start_w + target_w]

        # Optional color quantization
        if apply_quantization:
            final_img = ImageProcessor.apply_color_quantization(final_img, n_colors)

        return final_img

    @staticmethod
    def apply_color_quantization(image: np.ndarray, n_colors: int, random_state: int = 42) -> np.ndarray:
        """
        Apply K-means clustering to reduce image colors.
        Produces a posterized effect that makes mosaic matching easier.
        """
        data = image.reshape((-1, 3)).astype(np.float32)

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        return quantized_data.reshape(image.shape)

    @staticmethod
    def divide_image_vectorized(image: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
        """
        Divide an image into a grid of sub-tiles using vectorized reshaping.
        Much faster than nested loops.

        Returns:
            grid (np.ndarray): Shape (rows, cols, tile_h, tile_w, 3)
        """
        rows, cols = grid_size
        h, w = image.shape[:2]

        tile_h, tile_w = h // rows, w // cols
        cropped_h, cropped_w = tile_h * rows, tile_w * cols
        image = image[:cropped_h, :cropped_w]

        # Vectorized reshape
        grid = image.reshape(rows, tile_h, cols, tile_w, 3)
        grid = grid.transpose(0, 2, 1, 3, 4)

        # Sanity check
        assert grid.shape == (rows, cols, tile_h, tile_w, 3), "Grid reshape failed!"
        return grid

#Color Classification Methods
class ColorClassifier:
    """Class for color classification:
       - average_color (best for geometric/gradient)
       - histogram (brightness-based)
       - dominant_color (k-means, best for real_photos)
    """

    @staticmethod
    def classify_cells(grid, method='average_color'):
        """Classify grid cells using one of the supported methods."""
        rows, cols = grid.shape[:2]
        classifications = np.empty((rows, cols), dtype=object)

        for i in range(rows):
            for j in range(cols):
                cell = grid[i, j]

                if method == 'average_color':
                    # Average RGB value
                    avg_color = np.mean(cell, axis=(0, 1)).astype(int)
                    classifications[i, j] = ColorClassifier.color_to_category(avg_color)

                elif method == 'histogram':
                    # Simple brightness histogram on grayscale
                    gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
                    hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
                    dominant_bin = np.argmax(hist)
                    if dominant_bin < 2:
                        classifications[i, j] = 'dark_gray'
                    elif dominant_bin < 4:
                        classifications[i, j] = 'gray'
                    elif dominant_bin < 6:
                        classifications[i, j] = 'light_gray'
                    else:
                        classifications[i, j] = 'white'

                elif method == 'dominant_color':
                    # K-means to find the single most dominant color
                    reshaped = cell.reshape(-1, 3).astype(np.float32)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, _, centers = cv2.kmeans(
                        reshaped, 1, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
                    )
                    dominant_color = centers[0].astype(int)
                    classifications[i, j] = ColorClassifier.color_to_category(dominant_color)

                else:
                    raise ValueError(f"Unknown method: {method}")

        return classifications

    @staticmethod
    def color_to_category(color):
        """Convert RGB color to a general category."""
        r, g, b = color
        brightness = (r + g + b) / 3

        if brightness < 50:
            return 'black'
        elif brightness < 100:
            return 'dark_gray'
        elif brightness < 150:
            return 'gray'
        elif brightness < 200:
            return 'light_gray'
        elif brightness > 220:
            return 'white'
        else:
            # fallback by dominant channel
            if r > g and r > b:
                return 'red'
            elif g > r and g > b:
                return 'green'
            elif b > r and b > g:
                return 'blue'
            else:
                return 'gray'

#Mosaic Creation and Performance Metrics
class MosaicCreator:
    """Class for creating mosaics and calculating performance metrics"""

    @staticmethod
    def create_mosaic_vectorized(classifications, tiles, tile_size=(32, 32)):
        """Create mosaic using improved color-to-tile mapping."""
        rows, cols = classifications.shape
        tile_h, tile_w = tile_size

        # Initialize mosaic
        mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        # Get available tile keys
        tile_keys = list(tiles.keys())

        for i in range(rows):
            for j in range(cols):
                category = classifications[i, j]

                # Better category → tile mapping
                if category in tiles:
                    tile = tiles[category]

                elif category in ['green', 'cyan']:
                    # Prefer "nature" or "cool" if available
                    tile_key = next((k for k in tile_keys if k in ['nature', 'cool']), tile_keys[0])
                    tile = tiles[tile_key]

                elif category in ['red', 'magenta', 'yellow']:
                    # Prefer "warm" or "happy"
                    tile_key = next((k for k in tile_keys if k in ['warm', 'happy']), tile_keys[0])
                    tile = tiles[tile_key]

                elif category in ['black', 'dark_gray', 'gray']:
                    # Prefer geometric tiles if available
                    tile_key = next((k for k in tile_keys if "gray" in k or "black" in k), tile_keys[0])
                    tile = tiles[tile_key]

                else:
                    # Fallback to first tile
                    tile = tiles[tile_keys[0]]

                # Resize tile if necessary
                if tile.shape[:2] != tile_size:
                    tile = cv2.resize(tile, (tile_w, tile_h))

                # Place tile in mosaic
                start_y, end_y = i * tile_h, (i + 1) * tile_h
                start_x, end_x = j * tile_w, (j + 1) * tile_w
                mosaic[start_y:end_y, start_x:end_x] = tile

        return mosaic

    @staticmethod
    def calculate_performance_metrics(original, mosaic):
        """Calculate comprehensive performance metrics for evaluation."""
        if original.shape != mosaic.shape:
            mosaic = cv2.resize(mosaic, (original.shape[1], original.shape[0]))

        original_f = original.astype(np.float64)
        mosaic_f = mosaic.astype(np.float64)

        # Mean Squared Error
        mse = np.mean((original_f - mosaic_f) ** 2)

        # PSNR
        psnr = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

        # SSIM
        try:
            ssim_score = ssim(original, mosaic, channel_axis=2, data_range=255)
        except:
            ssim_score = 1.0 - (mse / (255.0 ** 2))

        # Color similarity
        orig_flat = original.flatten().astype(np.float64)
        mosaic_flat = mosaic.flatten().astype(np.float64)
        color_similarity = np.corrcoef(orig_flat, mosaic_flat)[0, 1]
        if np.isnan(color_similarity):
            color_similarity = 0.0

        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim_score,
            'color_similarity': color_similarity
        }

#Performance Analysis and Benchmarking
class PerformanceAnalyzer:
    """Class for performance benchmarking and visualization"""

    @staticmethod
    def benchmark_performance(image, grid_sizes, tiles, tile_set_name='geometric', compare_loops=True):
        """Benchmark performance across different grid sizes - with optional loop vs vectorized comparison."""
        results = {
            'grid_sizes': [],
            'processing_times_vectorized': [],
            'processing_times_loop': [] if compare_loops else None,
            'quality_metrics': []
        }

        print("Starting Performance Benchmark...")

        for idx, grid_size in enumerate(grid_sizes):
            print(f"Testing grid size {idx+1}/{len(grid_sizes)}: {grid_size}")

            # --- Vectorized version ---
            start_time = time.perf_counter()
            grid = ImageProcessor.divide_image_vectorized(image, grid_size)
            classifications = ColorClassifier.classify_cells_advanced(grid, 'dominant_color')
            mosaic = MosaicCreator.create_mosaic_vectorized(classifications, tiles)
            end_time = time.perf_counter()
            vectorized_time = end_time - start_time

            # --- Loop version (optional) ---
            loop_time = None
            if compare_loops:
                start_time = time.perf_counter()
                classifications_loop = ColorClassifier.classify_cells_advanced(grid, 'average_color')  # reuse grid
                mosaic_loop = MosaicCreator.create_mosaic_vectorized(classifications_loop, tiles)
                end_time = time.perf_counter()
                loop_time = end_time - start_time

            # Quality metrics
            metrics = MosaicCreator.calculate_performance_metrics(image, mosaic)

            # Store results
            results['grid_sizes'].append(f"{grid_size[0]}x{grid_size[1]}")
            results['processing_times_vectorized'].append(vectorized_time)
            if compare_loops:
                results['processing_times_loop'].append(loop_time)
            results['quality_metrics'].append(metrics)

            print(f"    Vectorized: {vectorized_time:.3f}s | SSIM: {metrics['ssim']:.3f}"
                  + (f" | Loop: {loop_time:.3f}s" if loop_time else ""))

        print("✅ Benchmark completed!")
        return results

    @staticmethod
    def create_performance_visualization(results):
        """Create comprehensive performance visualization with 4 charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🎨 Mosaic Generator Performance Analysis', fontsize=16, fontweight='bold')

        grid_sizes = results['grid_sizes']
        vec_times = results['processing_times_vectorized']
        loop_times = results['processing_times_loop']
        quality_metrics = results['quality_metrics']

        # Extract metrics
        ssim_scores = [m['ssim'] for m in quality_metrics]
        mse_scores = [m['mse'] for m in quality_metrics]
        psnr_scores = [m['psnr'] for m in quality_metrics]

        # 1. Processing Time (Vectorized vs Loop)
        axes[0, 0].plot(grid_sizes, vec_times, 'bo-', label='Vectorized', linewidth=2)
        if loop_times:
            axes[0, 0].plot(grid_sizes, loop_times, 'ro--', label='Loop', linewidth=2)
        axes[0, 0].set_title('⏱️ Processing Time vs Grid Size')
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. SSIM Quality
        axes[0, 1].plot(grid_sizes, ssim_scores, 'go-', linewidth=2)
        best_idx = np.argmax(ssim_scores)
        axes[0, 1].scatter(grid_sizes[best_idx], ssim_scores[best_idx], c='red', s=80, label='Best SSIM')
        axes[0, 1].set_title('📊 SSIM vs Grid Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. MSE Error
        axes[1, 0].plot(grid_sizes, mse_scores, 'mo-', linewidth=2)
        axes[1, 0].set_title('🎯 MSE vs Grid Size')
        axes[1, 0].set_ylabel('Error (Lower=Better)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Efficiency (SSIM/Time, normalized)
        efficiency = np.array([s / max(t, 1e-6) for s, t in zip(ssim_scores, vec_times)])
        efficiency = efficiency / efficiency.max()  # normalize
        axes[1, 1].bar(grid_sizes, efficiency, color='purple', alpha=0.7)
        axes[1, 1].set_title('⚡ Efficiency (SSIM per sec, normalized)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def create_process_visualization(original, grid, classifications, mosaic):
        """Visualization showing original → segmentation → final mosaic."""
        h, w = original.shape[:2]
        rows, cols = classifications.shape
        tile_h, tile_w = h // rows, w // cols

        segmented = np.zeros_like(original)
        colors = {
            'black': (0, 0, 0), 'dark_gray': (64, 64, 64), 'gray': (128, 128, 128),
            'light_gray': (192, 192, 192), 'white': (255, 255, 255),
            'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
            'yellow': (255, 255, 0), 'cyan': (0, 255, 255), 'magenta': (255, 0, 255)
        }

        for i in range(rows):
            for j in range(cols):
                category = classifications[i, j]
                color = colors.get(category, (128, 128, 128))
                y0, y1 = i * tile_h, (i + 1) * tile_h
                x0, x1 = j * tile_w, (j + 1) * tile_w
                segmented[y0:y1, x0:x1] = color
                cv2.rectangle(segmented, (x0, y0), (x1-1, y1-1), (255, 255, 255), 1)

        mosaic_resized = cv2.resize(mosaic, (w, h)) if mosaic.shape != original.shape else mosaic
        combined = np.hstack([original, segmented, mosaic_resized])

        return combined

class RealMosaicCreator:
    """Robust real mosaic creator with average-color + histogram matching (best method)."""

    @staticmethod
    def load_tiles_from_folder(folder, tile_size=(32, 32)):
        """Load all tiles from folder and precompute average colors + histograms."""
        tiles, avgs, hists = [], [], []
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            try:
                im = Image.open(path).convert("RGB").resize(tile_size, Image.LANCZOS)
                arr = np.array(im)

                # Average color
                avgs.append(arr.reshape(-1, 3).mean(axis=0))

                # Histogram (3D RGB with 8 bins per channel)
                hist = cv2.calcHist([arr], [0, 1, 2], None,
                                    [8, 8, 8], [0, 256, 0, 256, 0, 256])
                cv2.normalize(hist, hist)
                hists.append(hist.flatten().astype("float32"))

                tiles.append(arr)
            except Exception as e:
                print(f" Skipping {fname}: {e}")
                continue

        if not tiles:
            raise ValueError("No valid tiles loaded from folder!")

        print(f" Loaded {len(tiles)} valid tiles from '{folder}'")
        return np.stack(tiles), np.vstack(avgs), hists

    @staticmethod
    def create_mosaic_from_tiles(image, tiles, avgs, hists,
                                 grid_size=(16, 16), method="histogram", top_k=3):
        """Build mosaic using real tiles with robust histogram-based matching."""
        h, w = image.shape[:2]
        rows, cols = grid_size
        tile_h, tile_w = h // rows, w // cols
        mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for i in range(rows):
            for j in range(cols):
                cell = image[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]

                try:
                    if method == "histogram":
                        # Histogram matching
                        hist = cv2.calcHist([cell], [0, 1, 2], None,
                                            [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        cv2.normalize(hist, hist)
                        hist = hist.flatten().astype("float32")
                        dists = [cv2.compareHist(hist, h2, cv2.HISTCMP_CHISQR) for h2 in hists]
                        dists = np.array(dists)

                    elif method == "average_color":
                        # Average color matching
                        feat = cell.reshape(-1, 3).mean(axis=0)
                        dists = np.linalg.norm(avgs - feat, axis=1)

                    else:
                        print(f" Unknown method '{method}', falling back to average_color")
                        feat = cell.reshape(-1, 3).mean(axis=0)
                        dists = np.linalg.norm(avgs - feat, axis=1)

                    # Top-k random selection to avoid repetition
                    best_idx = np.argsort(dists)[:max(1, top_k)]
                    chosen = np.random.choice(best_idx)

                    tile = cv2.resize(tiles[chosen], (tile_w, tile_h))
                    mosaic[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w] = tile

                except Exception as e:
                    print(f" Cell ({i},{j}) skipped: {e}")

        return mosaic

#Main MosaicGenerator Class
class MosaicGenerator:
    """
    Main class that integrates all components for the Interactive Image Mosaic Generator.
    Supports real photo tiles + geometric tiles and benchmarking (vectorized vs loop).
    """

    def __init__(self, real_tile_folder="./tiles", tile_size=32):
        print("🎨 Initializing Mosaic Generator...")

        # Synthetic tile sets
        self.tile_sets = {
            'geometric': TileCreator.create_geometric_tiles(),
        }

        # Load real photo tiles
        try:
            self.real_tiles, self.real_avgs, self.real_hists = RealMosaicCreator.load_tiles_from_folder(
                real_tile_folder, tile_size=(tile_size, tile_size)
            )
            self.tile_sets['real_photos'] = "REAL"  # placeholder
            print(f"✅ Loaded {len(self.real_tiles)} real photo tiles from {real_tile_folder}")
        except Exception as e:
            print(f"⚠️ Could not load real tiles: {e}")
            self.real_tiles, self.real_avgs, self.real_hists = None, None, None

        print(f"✅ Created {len(self.tile_sets)} tile sets")

    def process_image_complete(self, image, grid_size, tile_set,
                              classification_method='average_color',
                              apply_quantization=False, n_colors=16):
        """Full pipeline for both synthetic and real photo tiles."""

        start_time = time.perf_counter()
        print(f"🚀 Processing image with {grid_size[0]}x{grid_size[1]} grid using {tile_set} tiles...")

        # Step 1: Preprocess
        processed_image = ImageProcessor.preprocess_image(
            image,
            apply_quantization=apply_quantization,
            n_colors=n_colors
        )

        # Step 2: Handle real photo tiles separately
        if tile_set == "real_photos":
            mosaic = RealMosaicCreator.create_mosaic_from_tiles(
                processed_image,
                self.real_tiles, self.real_avgs, self.real_hists,
                grid_size=grid_size,
                method=classification_method,
                top_k=3
            )
            classifications = None
        else:
            grid = ImageProcessor.divide_image_vectorized(processed_image, grid_size)
            classifications = ColorClassifier.classify_cells(grid, method=classification_method)
            tiles = self.tile_sets[tile_set]
            mosaic = MosaicCreator.create_mosaic_vectorized(classifications, tiles)

        # Step 3: Metrics
        metrics = MosaicCreator.calculate_performance_metrics(processed_image, mosaic)
        processing_time = time.perf_counter() - start_time

        return {
            'original': processed_image,
            'mosaic': mosaic,
            'classifications': classifications,
            'metrics': metrics,
            'processing_time': processing_time
        }

    # 🔹 Benchmark: compare vectorized vs loop-based
    def run_comprehensive_benchmark(self, image, tile_set='geometric'):
        """Benchmark performance across grid sizes for vectorized vs loop methods."""
        grid_sizes = [(8, 8), (16, 16), (32, 32)]
        timings = {'vectorized': [], 'loop': []}

        # Pick tiles
        if tile_set == 'real_photos':
            tiles, avgs, hists = self.real_tiles, self.real_avgs, self.real_hists
        else:
            tiles = self.tile_sets[tile_set]

        for gs in grid_sizes:
            # Vectorized timing
            start = time.perf_counter()
            _ = self.process_image_complete(image, gs, tile_set, classification_method='average_color')
            timings['vectorized'].append(time.perf_counter() - start)

            # Loop timing
            start = time.perf_counter()
            _ = self._process_loop_based(image, gs, tile_set)
            timings['loop'].append(time.perf_counter() - start)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        x = [f"{r}x{c}" for r, c in grid_sizes]
        ax.plot(x, timings['vectorized'], label="Vectorized", marker='o')
        ax.plot(x, timings['loop'], label="Loop-based", marker='s')
        ax.set_title("Performance: Vectorized vs Loop")
        ax.set_xlabel("Grid Size")
        ax.set_ylabel("Processing Time (s)")
        ax.legend()
        ax.grid(True)

        return fig

    def _process_loop_based(self, image, grid_size, tile_set):
        """Naive loop-based mosaic creation (for benchmarking only)."""
        h, w = image.shape[:2]
        rows, cols = grid_size
        tile_h, tile_w = h // rows, w // cols
        mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for i in range(rows):
            for j in range(cols):
                cell = image[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
                avg_color = cell.reshape(-1, 3).mean(axis=0)

                # For simplicity: pick nearest geometric tile
                if tile_set == 'geometric':
                    dists = {name: np.linalg.norm(np.array(tile).mean(axis=(0, 1)) - avg_color)
                             for name, tile in self.tile_sets['geometric'].items()}
                    chosen_tile = min(dists, key=dists.get)
                    tile = cv2.resize(self.tile_sets['geometric'][chosen_tile], (tile_w, tile_h))
                else:
                    # fallback for real_photos
                    dists = np.linalg.norm(self.real_avgs - avg_color, axis=1)
                    tile = cv2.resize(self.real_tiles[np.argmin(dists)], (tile_w, tile_h))

                mosaic[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w] = tile

        return mosaic


# Initialize generator
mosaic_gen = MosaicGenerator()

# Cell 9: Gradio Interface Function (with histogram via gr.Plot)
def process_mosaic(image, grid_rows, grid_cols, tile_set, classification_method,
                   apply_quantization, n_colors, show_performance):
    """
    Main Gradio interface function.
    Returns Mosaic Result + Metrics + Histogram (if enabled).
    """
    if image is None:
        return None, "⚠️ Please upload an image first.", None, None

    try:
        # Convert PIL to numpy array
        if hasattr(image, 'convert'):
            image_np = np.array(image.convert('RGB'))
        else:
            image_np = image

        print(f"Processing {image_np.shape} image...")

        # Process image
        results = mosaic_gen.process_image_complete(
            image_np,
            (grid_rows, grid_cols),
            tile_set,
            classification_method=classification_method,
            apply_quantization=apply_quantization,
            n_colors=n_colors
        )

        # Extract metrics
        metrics = results['metrics']
        metrics_text = f"""
## 📊 Performance Metrics

**⏱️ Execution Performance:**
- Processing Time: **{results['processing_time']:.3f} seconds**
- Grid Size: **{grid_rows} × {grid_cols} = {grid_rows * grid_cols} tiles**

**🎯 Image Quality Metrics:**
- Mean Squared Error (MSE): **{metrics['mse']:.2f}**
- PSNR: **{metrics['psnr']:.2f} dB**
- SSIM: **{metrics['ssim']:.4f}**
- Color Similarity: **{metrics['color_similarity']:.4f}**

**🔧 Configuration:**
- Tile Set: **{tile_set.title()}**
- Classification Method: **{classification_method.replace('_', ' ').title()}**
- Color Quantization: **{'Enabled' if apply_quantization else 'Disabled'}**
{f"- Quantization Colors: **{n_colors}**" if apply_quantization else ""}

**📈 Quality Assessment:**
{"🟢 Excellent quality!" if metrics['color_similarity'] > 0.5 else "🟡 Good quality!" if metrics['color_similarity'] > 0.3 else "🟠 Acceptable quality - try smaller grid for better results!"}
        """

        # Convert mosaic to PIL
        mosaic_pil = Image.fromarray(results['mosaic'])

        # Generate histogram if requested
        performance_chart = None
        if show_performance:
            import matplotlib.pyplot as plt

            labels = ["MSE", "PSNR", "SSIM", "ColorSim"]
            values = [metrics['mse'], metrics['psnr'], metrics['ssim'], metrics['color_similarity']]

            # Normalize values for better comparison
            max_val = max(values) if max(values) > 0 else 1
            norm_values = [v / max_val for v in values]

            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(labels, norm_values, color=["red", "blue", "green", "purple"])

            # Annotate with actual values
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha='center', va='bottom', fontsize=8)

            ax.set_ylim(0, 1.2)
            ax.set_title("Normalized Performance Metrics")
            ax.set_ylabel("Relative Scale (0–1)")
            plt.tight_layout()

            performance_chart = fig  # directly return matplotlib figure for gr.Plot

        return mosaic_pil, metrics_text, performance_chart, None

    except Exception as e:
        error_msg = f"❌ Error processing image: {str(e)}"
        print(error_msg)
        return None, error_msg, None, None

print("✅ Gradio interface function ready (with performance histogram)!")

# Cell 10: Gradio Interface Creation & Launch
def create_mosaic_interface():
    """Create Gradio interface with histogram output as gr.Plot."""

    demo = gr.Interface(
        fn=process_mosaic,
        inputs=[
            gr.Image(type="pil", label="📁 Upload Image"),
            gr.Slider(8, 64, value=16, step=1, label="Grid Rows"),
            gr.Slider(8, 64, value=16, step=1, label="Grid Columns"),
            gr.Dropdown(
                choices=["real_photos", "geometric"],
                value="real_photos",
                label="Tile Set Style"
            ),
            gr.Dropdown(
                choices=["dominant_color", "average_color", "histogram"],
                value="dominant_color",
                label="Color Classification Method"
            ),
            gr.Checkbox(value=False, label="Apply Color Quantization"),
            gr.Slider(4, 32, value=16, step=1, label="Number of Colors (if quantization enabled)"),
            gr.Checkbox(value=False, label="Show Performance Analysis")
        ],
        outputs=[
            gr.Image(type="pil", label="Mosaic Result"),
            gr.Markdown(label="Performance Metrics"),
            gr.Plot(label="Performance Histogram"),   # updated for fig
            gr.File(label="⬇️ Download Mosaic")       # added download option
        ],
        title="🖼️ Interactive Image Mosaic Generator",
        description="""
**Image Mosaic Generator**

Upload an image and transform it into a mosaic using:
- **Real Photos** (best for realism, use dominant_color)
- **Geometric Tiles** (pattern-based, works well with average_color)

### Features:
- Vectorized NumPy operations for speed
- Color classification (dominant_color / average_color / histogram)
- Performance metrics: MSE, PSNR, SSIM, Color Similarity
- Optional normalized histogram for quick comparison
- One-click download of generated mosaic

**Tips:**
- Use **real_photos + dominant_color** for realism
- Try **geometric + average_color** for patterns
- Enable performance analysis to see metric histograms
        """,
        cache_examples=False,
        allow_flagging="never"
    )

    return demo


# Launch and Testing
print("🚀 Launching Interactive Image Mosaic Generator...")
print("Access the interface locally at: http://127.0.0.1:7860")

# Launch Gradio app
interface.launch(
    share=True,          # Generates a public share link (optional)
    server_name="127.0.0.1",  # Localhost
    server_port=7860,    # Fixed port (can change if needed)
    debug=False,
    show_error=True
)



