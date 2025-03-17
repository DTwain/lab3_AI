from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json
import os
import sys
import time
import requests
import torch
from torchmetrics.text import CharErrorRate, WordErrorRate
import Levenshtein

def text_detection(image_url):
    """
    Return type : string
    image_url (string): A valid URL pointing to an image.
    Returns the text extracted from the image at the provided URL using Azure OCR.
    """
    subscription_key = os.environ["VISION_KEY"]
    endpoint = os.environ["VISION_ENDPOINT"]

    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    response = requests.get(image_url, stream=True)
    response.raise_for_status()  # Ensure we actually got the file
    image = Image.open(response.raw).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Call API with URL and raw response (allows you to get the operation location)
    read_response = computervision_client.read(image_url,  raw=True)

    # Get the operation location (URL with an ID at the end) from the response
    read_operation_location = read_response.headers["Operation-Location"]

    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1] 

    # Call the "GET" API and wait for it to retrieve the results 
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    ocr_output = ""
    recognized_boxes = []
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                box = line.bounding_box
                coords = [(box[i], box[i+1]) for i in range(0, len(box), 2)]
                draw.line(coords + [coords[0]], fill="red", width=1)
                ocr_output += line.text + "\n"
                recognized_boxes.append(box)

    return ocr_output, recognized_boxes, image

def plot_image(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def character_error_rate(image_text, detected_text):
    """
    Return type: float
    image_text (string) : The ground truth text from the image
    detected_text (string) : The text extracted from the image

    Character Error Rate (CER) using TorchMetrics.
    The result is in [0,1]: 0 means perfect match, 1 means 100% error rate.
    """
    cer_metric = CharErrorRate()

    references = [image_text]
    predictions = [detected_text]

    cer_value = cer_metric(predictions, references)
    return float(cer_value)

def word_error_rate(image_text, detected_text):
    """
    Return type: float
    image_text (string) : The ground truth text from the image
    detected_text (string) : The text extracted from the image

    Word Error Rate (WER) using TorchMetrics.
    The result is in [0,1]: 0 means perfect match, 1 means 100% error rate.
    """
    wer_metric = WordErrorRate()

    references = [image_text]
    predictions = [detected_text]

    wer_value = wer_metric(predictions, references)
    return float(wer_value)

def jaro_winkler_similarity(ground_truth, recognized_text):
    """
    Return type: float
    image_text (string) : The ground truth text from the image
    detected_text (string) : The text extracted from the image

    Jaro–Winkler similarity from python-Levenshtein, if installed.
    Range is [0,1], 1 means identical strings.
    """
    if Levenshtein is None:
        return None 
    return Levenshtein.jaro_winkler(ground_truth, recognized_text)

def hamming_distance(str_a, str_b):
    """
    Returns the Hamming Distance between two equal-length strings.
    If lengths differ, we'll raise ValueError (or handle it as needed).
    """
    if len(str_a) != len(str_b):
        raise ValueError("Hamming Distance is only defined for strings of the same length.")
    
    distance = 0
    for ch1, ch2 in zip(str_a, str_b):
        if ch1 != ch2:
            distance += 1
    return distance

def hamming_similarity(str_a, str_b):
    """
    Returns the fraction of positions that match.
    A value in [0..1]. 1 = identical strings, 0 = all different.
    Raises ValueError if different lengths.
    """
    if len(str_a) != len(str_b):
        raise ValueError("Hamming Similarity is only defined for strings of the same length.")
    
    dist = hamming_distance(str_a, str_b)
    return 1 - (dist / len(str_a))

def box_to_minmax(box):
    """
    box is [x1, y1, x2, y2, x3, y3, x4, y4].
    Returns (min_x, min_y, max_x, max_y) as a simple rectangle.
    """
    xs = [box[0], box[2], box[4], box[6]]
    ys = [box[1], box[3], box[5], box[7]]
    return (min(xs), min(ys), max(xs), max(ys))

def iou(rectA, rectB):
    """
    Compute Intersection over Union of two axis-aligned rectangles.
    rectA, rectB are (min_x, min_y, max_x, max_y).
    """
    (Ax1, Ay1, Ax2, Ay2) = rectA
    (Bx1, By1, Bx2, By2) = rectB

    # Calculate overlap
    overlap_x1 = max(Ax1, Bx1)
    overlap_y1 = max(Ay1, By1)
    overlap_x2 = min(Ax2, Bx2)
    overlap_y2 = min(Ay2, By2)

    overlap_width = max(0, overlap_x2 - overlap_x1)
    overlap_height = max(0, overlap_y2 - overlap_y1)
    overlap_area = overlap_width * overlap_height

    # Calculate each rect area
    areaA = (Ax2 - Ax1) * (Ay2 - Ay1)
    areaB = (Bx2 - Bx1) * (By2 - By1)

    # IoU = overlap / (areaA + areaB - overlap)
    union_area = areaA + areaB - overlap_area
    if union_area == 0:
        return 0.0
    else:
        return overlap_area / union_area

def compare_boxes(predicted_boxes, ground_truth_boxes):
    """
    Compare predicted vs. ground-truth bounding boxes line by line.
    Returns a list of IoU values (one per line) plus an average IoU.
    """
    if len(predicted_boxes) != len(ground_truth_boxes):
        print(f" WARNING: Mismatch in line counts: predicted={len(predicted_boxes)}, ground_truth={len(ground_truth_boxes)}")
        return []

    iou_values = []
    for i, (pred_box, gt_box) in enumerate(zip(predicted_boxes, ground_truth_boxes)):
        pred_rect = box_to_minmax(pred_box)
        gt_rect   = box_to_minmax(gt_box)
        iou_val   = iou(pred_rect, gt_rect)
        iou_values.append(iou_val)

    return iou_values


if __name__ == "__main__":
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, entry in enumerate(data, start=1):
        image_url = entry["image_url"]

        ground_truth_lines = entry["ground_truth"]
        ground_truth_text = "\n".join(ground_truth_lines)
        ground_truth_boxes = entry.get("ground_truth_boxes", [])

        print(f"\n=== Image #{idx} ===")
        print(f"URL: {image_url}")

        recognized_text, predicted_boxes, image = text_detection(image_url)

        print("Ground truth:")
        print(ground_truth_text)
        print()
        print("Recognized text:")
        print(recognized_text)

        cer_val = character_error_rate(ground_truth_text, recognized_text)
        wer_val = word_error_rate(ground_truth_text, recognized_text)
        jw_val = jaro_winkler_similarity(ground_truth_text, recognized_text)

        print("\n=== ERROR METRICS ===")
        print(f"CER: {cer_val * 100:.2f}%")  # multiply by 100 to get percentage
        print(f"WER: {wer_val * 100:.2f}%")
        print(f"Jaro–Winkler Similarity: {jw_val * 100:.2f}%")
        if len(ground_truth_text) == len(recognized_text):
            hs_val = hamming_similarity(ground_truth_text, recognized_text)
            print("Hamming distance:", hs_val)
        else:
            print("Hamming distance not applicable (unequal lengths).")


        iou_vals = compare_boxes(predicted_boxes, ground_truth_boxes)
        if iou_vals:
            avg_iou = sum(iou_vals) / len(iou_vals)
            print("\n=== BOUNDING BOX METRICS ===")
            for i, val in enumerate(iou_vals, start=1):
                print(f" Line {i} IoU: {val:.2f}")
            print(f" Average IoU: {avg_iou:.2f}")
        else:
            print("\nNo bounding box comparison (line mismatch or no ground_truth_boxes).")

        plot_image(image)
        