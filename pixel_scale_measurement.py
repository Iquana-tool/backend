
import cv2
import numpy as np
import os
import glob
import math
import csv

# === CONFIGURATION ===
KNOWN_DISTANCE_MM = 5.0
PIXEL_ASPECT_RATIO = 1.0
UNIT = "mm"
IMAGE_FOLDER = "coral_dataset/test/images"
OUTPUT_CSV = "measurements.csv"

# === STATE ===
ref_points = []
img_copy = None
overlay_img = None
measurements = []

def click_handler(event, x, y, flags, param):
    global ref_points, img_copy, overlay_img

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(ref_points) < 2:
            ref_points.append((x, y))
            overlay_img = img_copy.copy()

            # Draw point
            cv2.circle(overlay_img, (x, y), 5, (200, 200, 200), -1)

            if len(ref_points) == 2:
                x1, y1 = ref_points[0]
                x2, y2 = ref_points[1]

                # Snap to straight horizontal or vertical
                if abs(x2 - x1) > abs(y2 - y1):
                    y2 = y1  # horizontal
                else:
                    x2 = x1  # vertical
                ref_points[1] = (x2, y2)

                # Draw visual line
                cv2.circle(overlay_img, (x2, y2), 5, (200, 200, 200), -1)
                cv2.line(overlay_img, ref_points[0], ref_points[1], (180, 180, 180), 2)

                # Show distance
                dx = (x2 - x1) * PIXEL_ASPECT_RATIO
                dy = y2 - y1
                dist = math.hypot(dx, dy)
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                cv2.putText(overlay_img, f"{dist:.2f}px", (mid_x + 10, mid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

def get_image_paths(folder):
    return sorted(glob.glob(os.path.join(folder, "*.*")))

def main():
    global img_copy, overlay_img, ref_points

    image_paths = get_image_paths(IMAGE_FOLDER)

    print("[INFO] Click two points to draw a scale (5 mm).")
    print("[INFO] Line auto-snaps to horizontal/vertical. Press ENTER to save, BACKSPACE to redraw, ESC to exit.")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] Cannot load {filename}")
            continue

        ref_points = []
        img_copy = img.copy()
        overlay_img = img.copy()

        cv2.namedWindow("Measure Scale")
        cv2.setMouseCallback("Measure Scale", click_handler)

        while True:
            cv2.imshow("Measure Scale", overlay_img)
            key = cv2.waitKey(1)

            if key == 27:  # ESC
                print("[EXIT] Measurement aborted.")
                cv2.destroyAllWindows()
                save_results()
                return

            elif key == 8:  # BACKSPACE to reset drawing
                print("[RESET] Drawing cleared.")
                ref_points.clear()
                overlay_img = img_copy.copy()

            elif key == 13 and len(ref_points) == 2:  # ENTER to confirm
                x1, y1 = ref_points[0]
                x2, y2 = ref_points[1]
                dx = (x2 - x1) * PIXEL_ASPECT_RATIO
                dy = y2 - y1
                dist_px = math.hypot(dx, dy)

                scale_x = KNOWN_DISTANCE_MM / abs(dx) if dx != 0 else None
                scale_y = KNOWN_DISTANCE_MM / abs(dy) if dy != 0 else None
                scale_x_str = f"{scale_x:.4f}" if scale_x else "N/A"
                scale_y_str = f"{scale_y:.4f}" if scale_y else "N/A"

                measurements.append({
                    "filename": filename,
                    "distance_px": round(dist_px, 2),
                    "known_distance_mm": KNOWN_DISTANCE_MM,
                    "pixel_aspect_ratio": PIXEL_ASPECT_RATIO,
                    "scale_x_mm_per_px": scale_x_str,
                    "scale_y_mm_per_px": scale_y_str,
                    "unit": UNIT
                })

                print(f"[SAVED] {filename}: {dist_px:.2f}px (X: {scale_x_str}, Y: {scale_y_str})")
                cv2.destroyAllWindows()
                break

    save_results()

def save_results():
    if not measurements:
        print("[WARN] No measurements to save.")
        return

    keys = list(measurements[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(measurements)

        # Mean statistics
        distances = [m["distance_px"] for m in measurements]
        scales_x = [float(m["scale_x_mm_per_px"]) for m in measurements if m["scale_x_mm_per_px"] != "N/A"]
        scales_y = [float(m["scale_y_mm_per_px"]) for m in measurements if m["scale_y_mm_per_px"] != "N/A"]

        mean_dist = np.mean(distances)
        mean_sx = np.mean(scales_x) if scales_x else 0
        mean_sy = np.mean(scales_y) if scales_y else 0

        writer.writerow({})
        writer.writerow({
            "filename": "MEAN",
            "distance_px": round(mean_dist, 2),
            "known_distance_mm": KNOWN_DISTANCE_MM,
            "pixel_aspect_ratio": PIXEL_ASPECT_RATIO,
            "scale_x_mm_per_px": round(mean_sx, 4),
            "scale_y_mm_per_px": round(mean_sy, 4),
            "unit": UNIT
        })

    print(f"[DONE] {len(measurements)} measurements saved to '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()
