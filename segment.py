from typing import List
import numpy as np
import torch
from torch import nn
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import cv2


class Segmentator:
    def __init__(self):
        self.processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

    def predict(self, image: np.ndarray):
        inputs = self.processor(images=image, return_tensors="pt")

        outputs = self.model(**inputs)

        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]

        return pred_seg.numpy()
    
    def get_mean_color(self, image, bbox: List):
        x1, y1, x2, y2 = bbox
        person_img = image[y1:y2, x1:x2]

        pred_seg = self.predict(person_img)

        mask = pred_seg > 0
        mask = np.expand_dims(mask, -1)
        masked_img = person_img*mask

        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
        hue_channel = masked_img[:, :, 0]
        mean_hue = hue_channel[mask.squeeze() > 0].mean()
        return mean_hue
    
    def get_dominant_color(self, image: np.ndarray, bbox: List, k: int = 3):
        x1, y1, x2, y2 = bbox
        
        person_img = image[y1:y2, x1:x2]

        pred_seg = self.predict(person_img)

        mask = pred_seg > 0
        
        masked_pixels = person_img[mask]

        if masked_pixels.shape[0] == 0:
            return None

        pixels = np.float32(masked_pixels)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        unique, counts = np.unique(labels, return_counts=True)
        
        dominant_cluster_index = unique[np.argmax(counts)]
        
        dominant_color_bgr = centers[dominant_cluster_index]
        
        return dominant_color_bgr.astype(int).tolist()
