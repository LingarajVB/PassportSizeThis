import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import os
import urllib.request
from rembg import remove, new_session

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
        
        # Initialize rembg session for better performance
        self.rembg_session = new_session()

    def process_image(self, input_data, output_size=(413, 531), remove_bg=True):
        """
        Initial processing: Background removal and cropping.
        Standard passport size.
        input_data can be a file path or a PIL Image.
        """
        if isinstance(input_data, str):
            img = Image.open(input_data)
        else:
            img = input_data.copy()
            
        img = ImageOps.exif_transpose(img)
        
        if remove_bg:
            img = self._remove_background(img)
        
        # Convert PIL to numpy for MediaPipe
        img_np = np.array(img.convert('RGB'))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.detections:
            return self._center_crop(img, output_size)

        detection = detection_result.detections[0]
        bbox = detection.bounding_box
        x, y, width, height = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

        # Refined passport padding for a centered face and generous headroom
        # To center the face, we use equal padding on top/bottom and sides
        # A multiplier of 1.5-2.0 provides plenty of headroom and context
        padding_v = int(height * 1.5) 
        padding_h = int(width * 1.2)

        w, h = img.size
        
        # Calculate ideal crop boundaries centered on the face
        face_center_x = x + width // 2
        face_center_y = y + height // 2
        
        left = max(0, face_center_x - padding_h)
        top = max(0, face_center_y - padding_v)
        right = min(w, face_center_x + padding_h)
        bottom = min(h, face_center_y + padding_v)

        face_crop = img.crop((left, top, right, bottom))
        return self._resize_and_fill(face_crop, output_size)

    def apply_studio_background(self, person_img, bg_color_hex):
        """
        Applies a realistic studio background with vibrant colors and edge blending.
        """
        if person_img.mode != 'RGBA':
            return person_img

        w, h = person_img.size
        
        # 1. Create vibrant background
        # Convert hex to RGB
        bg_rgb = tuple(int(bg_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Create a base background with a subtle radial gradient
        # We use a simpler gradient to ensure colors stay vibrant and true to selection
        bg_img = Image.new("RGB", (w, h), bg_rgb)
        bg_draw = np.array(bg_img).astype(float)
        
        # Center of the gradient (roughly where the face is)
        cx, cy = w // 2, h // 3
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Normalize distance - much subtler gradient to preserve color purity
        max_dist = np.sqrt(cx**2 + cy**2)
        # Gradient ranges from 1.05 (slightly brighter) to 0.9 (slightly darker)
        gradient = 1.05 - (dist_from_center / (max_dist * 2.0))
        gradient = np.clip(gradient, 0.85, 1.05)
        
        # Apply gradient to RGB channels
        for i in range(3):
            bg_draw[:, :, i] *= gradient
            
        # Ensure values stay in valid range
        bg_draw = np.clip(bg_draw, 0, 255)
        bg_img = Image.fromarray(bg_draw.astype(np.uint8))
        
        # 2. Refine the person's edges (Feathering)
        person_array = np.array(person_img)
        alpha = person_array[:, :, 3]
        
        # Smooth the alpha mask slightly for better blending
        alpha_pil = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius=0.3))
        alpha = np.array(alpha_pil)
        
        # 3. Apply subtle color spill (rim lighting)
        person_rgb = person_array[:, :, :3].astype(float)
        
        # Find the edges of the alpha mask for spill
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, kernel)
        edge_mask = (edge_mask.astype(float) / 255.0) * 0.1 # 10% intensity for subtle spill
        
        for i in range(3):
            person_rgb[:, :, i] = person_rgb[:, :, i] * (1 - edge_mask) + bg_rgb[i] * edge_mask
            
        # Combine everything back
        refined_person = np.dstack([np.clip(person_rgb, 0, 255).astype(np.uint8), alpha])
        refined_person_img = Image.fromarray(refined_person, 'RGBA')
        
        # Composite person onto the background
        final_img = bg_img.copy()
        final_img.paste(refined_person_img, (0, 0), mask=refined_person_img)
        
        return final_img

    def enhance_image(self, img, brightness=1.0, contrast=1.0, sharpness=1.0, color=1.0,
                       edge_smoothing=0.0, skin_smoothing=0.0, redeye_removal=0.0,
                       teeth_whitening=0.0, eye_brightening=0.0, white_balance=0.0,
                       shadow_highlight=0.0, vignette_removal=0.0):
        """
        Applies manual and automatic studio enhancements with intensity control (0.0-1.0).
        """
        # Convert to numpy for OpenCV operations
        img_array = np.array(img)
        
        # Auto White Balance (intensity controls strength)
        if white_balance > 0:
            img_array = self._apply_white_balance(img_array, white_balance)
        
        # Shadow/Highlight Recovery (intensity controls clip limit)
        if shadow_highlight > 0:
            img_array = self._apply_shadow_highlight(img_array, shadow_highlight)
        
        # Vignette Removal (intensity controls correction strength)
        if vignette_removal > 0:
            img_array = self._apply_vignette_removal(img_array, vignette_removal)
        
        # Convert back to PIL
        img = Image.fromarray(img_array)
        
        # Red-eye Removal (intensity controls detection sensitivity)
        if redeye_removal > 0:
            img = self._apply_redeye_removal(img, redeye_removal)
        
        # Edge Smoothing (intensity controls filter strength)
        if edge_smoothing > 0:
            img = self._apply_edge_smoothing(img, edge_smoothing)
        
        # Skin Smoothing (intensity controls smoothing amount)
        if skin_smoothing > 0:
            img = self._apply_skin_smoothing(img, skin_smoothing)
        
        # Teeth Whitening (intensity controls whitening strength)
        if teeth_whitening > 0:
            img = self._apply_teeth_whitening(img, teeth_whitening)
        
        # Eye Brightening (intensity controls brightening amount)
        if eye_brightening > 0:
            img = self._apply_eye_brightening(img, eye_brightening)
        
        # Brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
        
        # Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
        
        # Sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)

        # Color/Saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(color)
        
        return img

    def _apply_white_balance(self, img_array, intensity=1.0):
        """Apply automatic white balance using gray world assumption with intensity control."""
        result = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        # Scale correction by intensity
        scale = intensity * 1.1
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * scale)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * scale)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        # Blend based on intensity
        return np.clip(img_array * (1 - intensity) + result * intensity, 0, 255).astype(np.uint8)

    def _apply_shadow_highlight(self, img_array, intensity=1.0):
        """Recover details in shadows and highlights with intensity control."""
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE for shadow recovery with intensity-based clip limit
        clip_limit = 0.5 + (intensity * 3.0)  # 0.5 to 3.5
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Blend original and enhanced based on intensity
        l_blended = (l * (1 - intensity) + l_enhanced * intensity).astype(np.uint8)
        
        lab = cv2.merge([l_blended, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_vignette_removal(self, img_array, intensity=1.0):
        """Remove dark corners from image with intensity control."""
        rows, cols = img_array.shape[:2]
        
        # Create vignette mask
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows)
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel / kernel.max()
        
        # Invert mask to brighten corners, scaled by intensity
        correction = 0.5 * intensity
        mask = 1 - (mask * correction)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        result = (img_array * mask).astype(np.uint8)
        return result

    def _apply_edge_smoothing(self, img, intensity=1.0):
        """Smooth jagged edges from background removal with intensity control."""
        img_array = np.array(img)
        # Scale filter parameters by intensity
        d = int(3 + intensity * 7)  # 3 to 10
        sigma_color = int(20 + intensity * 80)  # 20 to 100
        sigma_space = int(20 + intensity * 80)  # 20 to 100
        smoothed = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)
        # Blend based on intensity
        result = (img_array * (1 - intensity) + smoothed * intensity).astype(np.uint8)
        return Image.fromarray(result)

    def _apply_skin_smoothing(self, img, intensity=1.0):
        """Apply subtle skin smoothing using bilateral filter with intensity control."""
        img_array = np.array(img)
        
        # Detect skin areas
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply bilateral filter with intensity-based strength
        d = int(5 + intensity * 10)  # 5 to 15
        sigma = int(30 + intensity * 70)  # 30 to 100
        smoothed = cv2.bilateralFilter(img_array, d, sigma, sigma)
        
        # Blend original and smoothed based on skin mask and intensity
        mask_3ch = np.repeat(skin_mask[:, :, np.newaxis], 3, axis=2) / 255.0
        mask_3ch = mask_3ch * intensity  # Scale by intensity
        result = (img_array * (1 - mask_3ch) + smoothed * mask_3ch).astype(np.uint8)
        
        return Image.fromarray(result)

    def _apply_redeye_removal(self, img, intensity=1.0):
        """Detect and remove red-eye effect with intensity control."""
        img_array = np.array(img)
        
        # Convert to HSV for better red detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Red color ranges (red wraps around in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Find circular red areas (potential eyes)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = img_array.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            if 10 < area < 500:  # Eye-sized areas
                x, y, w, h = cv2.boundingRect(contour)
                # Reduce red channel in the detected area, scaled by intensity
                roi = result[y:y+h, x:x+w]
                red_reduction = 1.0 - (0.3 * intensity)  # 1.0 to 0.7
                green_boost = 1.0 + (0.1 * intensity)  # 1.0 to 1.1
                blue_boost = 1.0 + (0.1 * intensity)  # 1.0 to 1.1
                roi[:, :, 0] = (roi[:, :, 0] * red_reduction).astype(np.uint8)
                roi[:, :, 1] = np.clip(roi[:, :, 1] * green_boost, 0, 255).astype(np.uint8)
                roi[:, :, 2] = np.clip(roi[:, :, 2] * blue_boost, 0, 255).astype(np.uint8)
                result[y:y+h, x:x+w] = roi
        
        return Image.fromarray(result)

    def _apply_teeth_whitening(self, img, intensity=1.0):
        """Subtle teeth whitening with intensity control."""
        img_array = np.array(img)
        
        # Convert to HSV
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Detect white/yellow areas (potential teeth)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([50, 80, 255])
        teeth_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply whitening to detected areas
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Increase lightness in teeth areas based on intensity
        mask_bool = teeth_mask > 0
        l = l.astype(float)
        lightness_mult = 1.0 + (0.05 * intensity)  # 1.0 to 1.05
        lightness_add = 5 * intensity  # 0 to 5
        l[mask_bool] = np.clip(l[mask_bool] * lightness_mult + lightness_add, 0, 255)
        l = l.astype(np.uint8)
        
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(result)

    def _apply_eye_brightening(self, img, intensity=1.0):
        """Brighten and enhance eyes with intensity control."""
        img_array = np.array(img)
        
        # Convert to grayscale for eye detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Load eye cascade classifier
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        if eye_cascade.empty():
            return img
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)
        
        result = img_array.copy()
        for (x, y, w, h) in eyes:
            # Define eye region
            eye_roi = result[y:y+h, x:x+w]
            
            # Apply CLAHE to eye region for brightening with intensity control
            lab = cv2.cvtColor(eye_roi, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clip_limit = 0.5 + (intensity * 2.5)  # 0.5 to 3.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
            l_enhanced = clahe.apply(l)
            # Blend based on intensity
            l = (l * (1 - intensity) + l_enhanced * intensity).astype(np.uint8)
            lab = cv2.merge([l, a, b])
            eye_roi = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Contrast boost based on intensity
            alpha = 1.0 + (0.1 * intensity)  # 1.0 to 1.1
            beta = 5 * intensity  # 0 to 5
            eye_roi = cv2.convertScaleAbs(eye_roi, alpha=alpha, beta=beta)
            
            result[y:y+h, x:x+w] = eye_roi
        
        return Image.fromarray(result)

    def _remove_background(self, img):
        """
        Removes background and returns RGBA image with transparency.
        """
        return remove(img, session=self.rembg_session, alpha_matting=True)

    def _resize_and_fill(self, img, size):
        """Resizes image to fill the target size exactly, preserving RGBA if present."""
        # Use ImageOps.fit but handle transparency correctly
        if img.mode == 'RGBA':
            # Create a transparent target
            res = Image.new("RGBA", size, (0, 0, 0, 0))
            fitted = ImageOps.fit(img, size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            res.paste(fitted, (0, 0), mask=fitted)
            return res
        return ImageOps.fit(img, size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    def _center_crop(self, img, size):
        """Fallback center crop if no face is detected, using ImageOps.fit to fill the frame."""
        return ImageOps.fit(img, size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))

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
