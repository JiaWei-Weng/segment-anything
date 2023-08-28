import os
import cv2
import numpy as np
import pandas as pd
import glob
import re
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import argparse

def process_images(layer_path, img_path, output_folder):
    # Load layer and image
    layer = cv2.imread(layer_path)
    img = cv2.imread(img_path)

    # Calculate overlap
    alpha = 0.5
    result = cv2.addWeighted(img, 1 - alpha, layer, alpha, 0)
    result_path = os.path.join(output_folder, "overlap.png")
    cv2.imwrite(result_path, result)

    # Extract mask
    extracted_part = np.zeros_like(img)
    extracted_part[layer == 255] = img[layer == 255]
    extracted_part_path = os.path.join(output_folder, "extract_img.png")
    cv2.imwrite(extracted_part_path, extracted_part)

    # Extract bounding box
    layer_gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(layer_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    # Extract regions based on bounding box
    extracted_part_2 = extracted_part[y:y+h+1, x:x+w+1]
    extracted_part_3 = img[y:y+h+1, x:x+w+1]

    extracted_part_path_2 = os.path.join(output_folder, "extract_img_2.png")
    extracted_part_path_3 = os.path.join(output_folder, "extract_img_3.png")

    cv2.imwrite(extracted_part_path_2, extracted_part_2)
    cv2.imwrite(extracted_part_path_3, extracted_part_3)

def merge_and_process_images(csv_folder, try_num):
    csv_files = glob.glob(os.path.join(csv_folder, "DSC_*", "metadata.csv"))
    print(f"csv_files: {len(csv_files)} files...")
    print("start merging...")

    merged_df = pd.DataFrame()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dsc_folder_value = re.search(r'\\(\w+)_output\\', csv_file).group(1)
        dsc_value = csv_file.split('\\')[-2]  # 获取 DSC_* 部分
        df['DSC_folder'] = dsc_folder_value
        df['DSC'] = dsc_value
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    print("metadata length: ", len(merged_df))
    print("metadata columns: ", merged_df.columns.tolist())

    # Clustering
    selected_column = ['area', 'bbox_x0', 'bbox_y0', 'bbox_w', 'bbox_h']
    selected_data = merged_df[selected_column]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)

    dbscan = DBSCAN(eps=0.3, min_samples=10)
    labels = dbscan.fit_predict(scaled_data)

    merged_df["labels"] = labels
    mask = merged_df["labels"] == 0

    for folder_name, dsc, id_name in merged_df[mask][["DSC_folder", "DSC", "id"]].values[:try_num]:
        parts_path = ["H:", "3Ddom", "segment_anything_backups", "data_output", "101ND750", "images_3_samauto_1", dsc]
        parts_path.append((str(id_name) + '.png'))
        layer_path = os.path.join('', *parts_path)

        parts_path = ["H:", "3Ddom", "segment_anything_backups", "data", "101ND750", "images_3"]
        parts_path.append((dsc + '.png'))
        img_path = os.path.join('', *parts_path)

        output_folder = os.path.join(".", "result_mask", dsc)

        os.makedirs(output_folder, exist_ok=True)

        # Save the original image
        original_img = cv2.imread(img_path)
        original_img_path = os.path.join(output_folder, "original_img.png")
        cv2.imwrite(original_img_path, original_img)

        process_images(layer_path, img_path, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images based on CSV files and perform clustering.")
    parser.add_argument("--csv-folder", required=True, help="Path to the CSV folder")
    parser.add_argument("--try-num", type=int, default=1, help="Number of runs to process")
    args = parser.parse_args()

    csv_folder = args.csv_folder
    try_num = args.try_num
    merge_and_process_images(csv_folder, try_num)
