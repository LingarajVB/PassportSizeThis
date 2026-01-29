import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageOps
import os
import urllib.request

class PassportGenerator:
    def __init__(self):
        # Download the model file if it doesn't exist
        model_path = 'detector.tflite'
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(url, model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)

    def process_image(self, input_path, output_size=(350, 450)):
        """
        Processes an image to create a passport size photo.
        """
        # Load image with PIL
        img = Image.open(input_path)
        img = ImageOps.exif_transpose(img)
        
        # Convert to Mediapipe Image
        mp_image = mp.Image.create_from_file(input_path)
        
        # Detect faces
        detection_result = self.detector.detect(mp_image)

        if not detection_result.detections:
            return self._center_crop(img, output_size)

        # Get the first face
        detection = detection_result.detections[0]
        bbox = detection.bounding_box
        
        # Mediapipe tasks bbox is in pixels: x, y, width, height
        x, y, width, height = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

        # Padding for passport look
        padding_y = int(height * 0.6)
        padding_x = int(width * 0.5)

        w, h = img.size
        left = max(0, x - padding_x)
        top = max(0, y - padding_y)
        right = min(w, x + width + padding_x)
        bottom = min(h, y + height + padding_y)

        face_crop = img.crop((left, top, right, bottom))
        return self._resize_and_fill(face_crop, output_size)

    def _resize_and_fill(self, img, size):
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Create a white background
        new_img = Image.new("RGB", size, (255, 255, 255))
        # Center the resized image
        offset = ((size[0] - img.size[0]) // 2, (size[1] - img.size[1]) // 2)
        new_img.paste(img, offset)
        return new_img

    def _center_crop(self, img, size):
        # Fallback center crop
        w, h = img.size
        target_w, target_h = size
        
        aspect_target = target_w / target_h
        aspect_img = w / h
        
        if aspect_img > aspect_target:
            # Image is wider than target
            new_w = int(h * aspect_target)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        else:
            # Image is taller than target
            new_h = int(w / aspect_target)
            top = (h - new_h) // 2
            img = img.crop((0, top, w, top + new_h))
            
        return img.resize(size, Image.Resampling.LANCZOS)

    def create_grid(self, passport_img, count=8):
        """
        Creates a grid of 8 or 16 images.
        8 images: 2x4 grid
        16 images: 4x4 grid
        """
        w, h = passport_img.size
        margin = 20
        
        if count == 8:
            cols, rows = 4, 2
        elif count == 16:
            cols, rows = 4, 4
        else:
            # Default to 8 if something else is passed
            cols, rows = 4, 2

        grid_w = (w * cols) + (margin * (cols + 1))
        grid_h = (h * rows) + (margin * (rows + 1))
        
        grid_img = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
        
        for r in range(rows):
            for c in range(cols):
                x = margin + c * (w + margin)
                y = margin + r * (h + margin)
                grid_img.paste(passport_img, (x, y))
                
        return grid_img
