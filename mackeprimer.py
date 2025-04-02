import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_images(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    sorted_files = []
    for f in image_files:
        underscore_index = f.index('_') + 1
        dot_index = f.index('.')
        number = int(f[underscore_index:dot_index])
        sorted_files.append((number, f))
    sorted_files.sort()
    images = {}
    for _, f in sorted_files:
        image_path = os.path.join(folder_path, f)
        image = cv2.imread(image_path) 
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images[f] = image_rgb

    return images

def load_cats(csv_path):
    actual_counts = {} 
    with open(csv_path, 'r') as file:
        lines = file.readlines() 
        for line in lines[1:]:   
            image_name, count = line.strip().split(',')
            actual_counts[image_name] = int(count)
    return actual_counts


def display_image_with_contours(image, contours):
     output_image = image.copy()
     if len(contours) > 0:
         cv2.drawContours(output_image, contours, -1, (255, 0, 0), 2)  
     plt.imshow(output_image)
     plt.axis('off')
     plt.show()

def create_combined_cat_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #StackOverflow-Identifying the range of a color in HSV using OpenCV
    color_ranges = {
        "brown": ((10, 70, 50), (20, 255, 255)),
        "orange": ((5, 150, 100), (15, 255, 255)),
        "gray": ((0, 0, 40), (180, 50, 180)),
        "light_gray_white": ((0, 0, 200), (180, 20, 255)),
        "blue_gray": ((80, 10, 20), (140, 150, 150)), 
        "orange_white": ((8, 13, 180), (25, 90, 255)),  
    }

    black_ranges = {"black": ((0, 0, 0), (180, 255, 60))
}
    combined_mask = np.zeros_like(hsv[:, :, 0])
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    black_mask = np.zeros_like(hsv[:, :, 0])
    for color, (lower, upper) in black_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        black_mask = cv2.bitwise_or(black_mask, mask)
    
    kernel_erode = np.ones((3, 3), np.uint8)   
    kernel_dilate = np.ones((5, 5), np.uint8)  
    black_mask = cv2.erode(black_mask, kernel_erode, iterations=2)
    black_mask = cv2.dilate(black_mask, kernel_dilate, iterations=3)
    final_mask = cv2.bitwise_or(combined_mask, black_mask)
    # plt.imshow(black_mask, cmap='gray')
    # plt.title("Black mask ")
    # plt.show()
    return final_mask


def detect_cats_with_contours(image):
    mask = create_combined_cat_mask(image)
    _, bin_img = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.dilate(bin_img, kernel, iterations=1) 
    bin_img = cv2.erode(bin_img, kernel, iterations=1) 
    
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 1000
    max_area = 17900
    valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    
    return valid_contours

def main(folder_path):
    images = load_images(folder_path)
    csv_path = os.path.join(folder_path, 'cat_count.csv')

    actual_counts = load_cats(csv_path)

    results = []

    for filename, image in images.items():
        contours = detect_cats_with_contours(image)
        display_image_with_contours(image, contours)
        num_cats = len(contours)
        true_count = actual_counts.get(filename, np.nan)
        results.append({'filename': filename, 'predicted': num_cats, 'actual': true_count})
        #print(f'Picture {filename}: detected {num_cats} cats.')

    total_error = sum(abs(r['predicted'] - r['actual']) for r in results)
    mae = total_error / len(results)    
    print(mae)


folder_path = sys.argv[1]
main(folder_path)
