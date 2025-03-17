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
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                box = line.bounding_box
                coords = [(box[i], box[i+1]) for i in range(0, len(box), 2)]
                draw.line(coords + [coords[0]], fill="red", width=1)
                ocr_output += line.text + "\n"

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    return ocr_output

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


if __name__ == "__main__":
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, entry in enumerate(data, start=1):
        image_url = entry["image_url"]

        ground_truth_lines = entry["ground_truth"]
        ground_truth_text = "\n".join(ground_truth_lines)

        print(f"\n=== Image #{idx} ===")
        print(f"URL: {image_url}")

        recognized_text = text_detection(image_url)

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