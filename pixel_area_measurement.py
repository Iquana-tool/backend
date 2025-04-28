
import cv2
import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt

# === CONFIGURATION ===
IMAGE_FOLDER = "polyp_dataset/test/images"
SCALE_CSV = "output/measurements.csv"
OUTPUT_CSV = "output/calculated_image_areas.csv"
UNIT = "mm"
SAVE_GRAPH = "output/area_graph.png"
MASK_OUTPUT_FOLDER = "output/output_masks"
MIN_CONTOUR_AREA = 5000  

# === FUNCTIONS ===

def load_mean_scale(csv_path):
    try:
        df = pd.read_csv(csv_path, sep=",", engine="python", on_bad_lines='skip')
    except Exception as e:
        print(f"[ERROR] Problem reading CSV: {e}")
        return None

    mean_row = df[df['filename'].str.upper() == 'MEAN']
    if mean_row.empty:
        print("[ERROR] MEAN row not found in CSV. Exiting.")
        return None

    scale_x = float(mean_row['scale_x_mm_per_px'].values[0])
    scale_y = float(mean_row['scale_y_mm_per_px'].values[0])
    print(f"[INFO] Loaded MEAN scale: ({scale_x}, {scale_y})")
    return scale_x, scale_y

def get_image_paths(folder):
    if not os.path.isdir(folder):
        print(f"[ERROR] Folder does not exist: {folder}")
        return []
    return sorted(glob.glob(os.path.join(folder, "*.*")))

def create_cleaned_mask(img, filename):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([10, 50, 50])
    upper = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphology to clean small noises
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Filter only big enough areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask)

    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Save cleaned mask
    os.makedirs(MASK_OUTPUT_FOLDER, exist_ok=True)
    cv2.imwrite(os.path.join(MASK_OUTPUT_FOLDER, filename), filtered_mask)

    return filtered_mask

def calculate_area(img_path, scale_x_mm_per_px, scale_y_mm_per_px):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read image: {img_path}")
        return None, None

    filename = os.path.basename(img_path)
    mask = create_cleaned_mask(img, filename)
    white_pixels = np.sum(mask == 255)
    area_mm2 = white_pixels * scale_x_mm_per_px * scale_y_mm_per_px
    return white_pixels, area_mm2

def save_results(results, output_csv):
    df = pd.DataFrame(results, columns=["filename", "pixel_count", f"area_{UNIT}²", "scale_x_mm_per_px", "scale_y_mm_per_px"])
    df[f"area_{UNIT}²"] = df[f"area_{UNIT}²"].round(2)
    df["scale_x_mm_per_px"] = df["scale_x_mm_per_px"].round(4)
    df["scale_y_mm_per_px"] = df["scale_y_mm_per_px"].round(4)
    df.to_csv(output_csv, index=False)
    print(f"\n Results saved to '{output_csv}'.")

def plot_area_graph(results):
    df = pd.DataFrame(results, columns=["filename", "pixel_count", f"area_{UNIT}²", "scale_x_mm_per_px", "scale_y_mm_per_px"])
    df = df.sort_values(by=f"area_{UNIT}²", ascending=False)

    plt.figure(figsize=(16, 8))
    plt.bar(df["filename"], df[f"area_{UNIT}²"], color='orange')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.xlabel('Image Filename', fontsize=12)
    plt.ylabel(f'Area ({UNIT}²)', fontsize=12)
    plt.title('Coral Area Measurement per Image', fontsize=14)
    plt.tight_layout()
    plt.savefig(SAVE_GRAPH)
    print(f" Graph saved as '{SAVE_GRAPH}'.")

def main():
    print("[INFO] Loading MEAN scale info...")
    mean_scale = load_mean_scale(SCALE_CSV)

    if mean_scale is None:
        return

    scale_x, scale_y = mean_scale

    print("[INFO] Reading images...")
    image_paths = get_image_paths(IMAGE_FOLDER)

    if not image_paths:
        print("[ERROR] No images found. Exiting.")
        return

    results = []

    for img_path in image_paths:
        filename = os.path.basename(img_path)

        white_pixels, area_mm2 = calculate_area(img_path, scale_x, scale_y)
        if white_pixels is not None:
            print(f"[OK] {filename}: {white_pixels} pixels -> {area_mm2:.2f} {UNIT}²")
            results.append((filename, white_pixels, area_mm2, scale_x, scale_y))

    if results:
        save_results(results, OUTPUT_CSV)
        plot_area_graph(results)
    else:
        print("[WARN] No valid images processed.")

    print(f"\n Summary:")
    print(f"  Total Images Processed: {len(results)}")

if __name__ == "__main__":
    main()
