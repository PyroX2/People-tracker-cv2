import numpy as np
import cv2

class KCFTracker:
    def __init__(self):
        self.lambdar = 0.0001
        self.sigma = 0.5
        self.output_sigma_factor = 0.125
        self.interp_factor = 0.02   
        self.padding = 1.5          
        
        # 1. FAIL SAFE THRESHOLD
        # If max correlation < 0.2, we assume loss. 
        # Increase this (e.g., 0.25) if you still see false positives.
        self.detect_thresh = 0.20   
        
        # 2. MOVEMENT CLAMP
        # If the object moves more than this many pixels in one frame, ignore it.
        # This prevents the "teleporting" to random spots.
        self.max_step = 60          
        
        self._roi = None
        self._alphaf = None
        self._xf = None
        self._hann = None
        self._template_size = [0, 0]

    def init(self, image, bbox):
        x, y, w, h = map(int, bbox)
        
        padded_w = int(w * (1 + self.padding))
        padded_h = int(h * (1 + self.padding))
        
        self._roi = [x, y, w, h]
        self._template_size = [padded_w, padded_h]

        # Create Hanning Window (Broadcastable to 3 channels)
        hann_rows = np.hanning(padded_h)
        hann_cols = np.hanning(padded_w)
        # Shape: (H, W) -> (H, W, 1) so we can multiply with (H, W, 3) image
        self._hann = np.outer(hann_rows, hann_cols)[..., None].astype(np.float32)

        # Extract patch (Returns 3-channel image now)
        patch = self._get_subwindow(image, self._roi, self._template_size)
        
        # Gaussian Target (Same as before, 2D)
        y = self._gaussian_peak(padded_w, padded_h)
        self._yf = np.fft.fft2(y)

        # Train with COLOR features
        # axes=(0,1) ensures we perform 2D FFT on each channel independently
        xf = np.fft.fft2(patch, axes=(0,1))
        kf = self._gaussian_correlation(xf, xf)
        
        self._alphaf = self._yf / (kf + self.lambdar)
        self._xf = xf
        return True

    def update(self, image):
        if self._roi is None: return False, (0,0,0,0)

        patch = self._get_subwindow(image, self._roi, self._template_size)
        zf = np.fft.fft2(patch, axes=(0,1))
        
        kf = self._gaussian_correlation(self._xf, zf)
        response_f = self._alphaf * kf
        response = np.fft.ifft2(response_f).real
        
        # --- FIX: ROBUST PEAK DETECTION ---
        
        # 1. Shift the response so (0,0) frequency is at the center of the array
        # This makes finding the delta (dx, dy) much more intuitive.
        response_shifted = np.fft.fftshift(response)
        
        max_val = np.max(response_shifted)
        
        # Fail if response is too weak
        if max_val < self.detect_thresh:
            return False, tuple(self._roi)

        # Find location of max
        max_r, max_c = np.unravel_index(np.argmax(response_shifted), response_shifted.shape)
        
        # Calculate displacement relative to center
        # Since we shifted, the center index represents 0 movement.
        center_y, center_x = response_shifted.shape[0] // 2, response_shifted.shape[1] // 2
        dy = max_r - center_y
        dx = max_c - center_x
        
        # Sanity Check: Prevent Teleportation
        if abs(dx) > self.max_step or abs(dy) > self.max_step:
            return False, tuple(self._roi)

        # Update ROI
        self._roi[0] += dx
        self._roi[1] += dy
        
        # Clip to image
        x, y, w, h = self._roi
        img_h, img_w = image.shape[:2]
        self._roi[0] = max(0, min(img_w - w, x))
        self._roi[1] = max(0, min(img_h - h, y))
        
        # Update Model
        new_patch = self._get_subwindow(image, self._roi, self._template_size)
        new_xf = np.fft.fft2(new_patch, axes=(0,1))
        kf_new = self._gaussian_correlation(new_xf, new_xf)
        new_alphaf = self._yf / (kf_new + self.lambdar)
        
        self._alphaf = (1 - self.interp_factor) * self._alphaf + self.interp_factor * new_alphaf
        self._xf = (1 - self.interp_factor) * self._xf + self.interp_factor * new_xf
        
        return True, tuple(self._roi)

    def _get_subwindow(self, image, roi, template_size):
        x, y, w, h = map(int, roi)
        cx, cy = x + w // 2, y + h // 2
        tw, th = template_size
        
        # Extract 3-Channel Patch
        patch = cv2.getRectSubPix(image, (tw, th), (cx, cy))
        
        # Normalize 0-1
        patch = patch.astype(np.float32) / 255.0
        
        # Hanning window is applied to all 3 channels
        return patch * self._hann

    def _gaussian_peak(self, w, h):
        output_sigma = np.sqrt(w * h) * self.output_sigma_factor / 0.1
        sy, sx = np.ogrid[-h//2:h//2, -w//2:w//2] # Grid centered at 0
        dist = sx**2 + sy**2
        response = np.exp(-0.5 * (dist / output_sigma**2))
        return np.fft.ifftshift(response) # Shift back to top-left for FFT compatibility

    def _gaussian_correlation(self, xf, yf):
        """
        Multichannel Gaussian Correlation
        Sum the dot products across all channels (axis 2)
        """
        N = xf.shape[0] * xf.shape[1] # Number of pixels in one channel
        
        # 1. Element-wise multiplication (Correlation in freq domain)
        # Sum across channels (axis=2) to merge Color features
        xyf = xf * np.conj(yf)
        xy_sum = np.sum(xyf, axis=2)
        xy = np.real(np.fft.ifft2(xy_sum))
        
        # 2. Auto-correlation terms (Sum of squares in spatial domain)
        # Parseval's theorem: sum(|x|^2) in time = 1/N * sum(|X|^2) in freq
        xx_sum = np.sum(np.abs(xf)**2, axis=2) / N
        yy_sum = np.sum(np.abs(yf)**2, axis=2) / N
        
        # We need the scalar sum of energies. 
        # In KCF simplified, we approximate ||x||^2 + ||y||^2 as constant or use the DC component
        # But rigorous way for Gaussian kernel:
        xx_val = np.sum(xx_sum) 
        yy_val = np.sum(yy_sum)
        
        # In the simplified KCF trick, we assume cyclic shifts have same energy.
        # The term ||x||^2 + ||y||^2 - 2 * x . y
        # We need xx and yy to be scalars (total energy of patch), xy is the 2D correlation map
        
        # Improved Kernel calculation for stability:
        # Re-calculate energies in spatial domain to be safe (Parseval is tricky with padding)
        # But sticking to the standard KCF open-source logic:
        
        xf_sq = np.real(np.fft.ifft2(np.sum(xf * np.conj(xf), axis=2)))
        yf_sq = np.real(np.fft.ifft2(np.sum(yf * np.conj(yf), axis=2)))
        
        # For a circular shift, the norm ||x||^2 is constant for all shifts.
        # So we just take the first element (zero shift) energy
        xx = xf_sq[0,0]
        yy = yf_sq[0,0]
        
        term = xx + yy - 2 * xy
        term = np.maximum(term, 0) / N # Normalize by pixels
        
        k = np.exp((-1 / (self.sigma**2)) * term)
        
        return np.fft.fft2(k)

def TrackerKCF_create():
    return KCFTracker()

import numpy as np
import cv2

class ReliabilityKCFTracker:
    def __init__(self):
        # --- KCF Parameters ---
        self.lambdar = 0.0001
        self.sigma = 0.5
        self.output_sigma_factor = 0.125
        self.interp_factor = 0.02   
        self.padding = 1.5
        self.cell_size = 1
        self.dead_zone = 0.07
        
        # --- RELIABILITY PARAMETERS (The CSRT-like part) ---
        # If the color match score drops below this, we stop LEARNING (prevent poisoning)
        self.learning_thresh = 0.4
        # If the color match score drops below this, we declare LOSS
        self.loss_thresh = 0.15
        
        self._roi = None
        self._alphaf = None
        self._xf = None
        self._hann = None
        self._template_size = [0, 0]
        
        # Histogram for Color Reliability
        self._hist = None

    def init(self, image, bbox):
        x, y, w, h = map(int, bbox)
        
        padded_w = int(w * (1 + self.padding))
        padded_h = int(h * (1 + self.padding))
        
        self._roi = [x, y, w, h]
        self._template_size = [padded_w, padded_h]

        # 1. Initialize Color Histogram (The "Reliability" Model)
        # We look strictly inside the bounding box (no padding) for color init
        roi_patch = image[y:y+h, x:x+w]
        if roi_patch.size == 0: return False
        
        # Convert to HSV to be robust against lighting changes
        hsv_roi = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2HSV)
        
        # Calculate Histogram (Hue and Saturation only, ignore Value/Brightness)
        # 16 bins for Hue, 16 for Saturation
        self._hist = cv2.calcHist([hsv_roi], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(self._hist, self._hist, 0, 255, cv2.NORM_MINMAX)

        # 2. Standard KCF Init
        hann_rows = np.hanning(padded_h)
        hann_cols = np.hanning(padded_w)
        self._hann = np.outer(hann_rows, hann_cols)[..., None].astype(np.float32)

        patch = self._get_subwindow(image, self._roi, self._template_size)
        
        # Gaussian Target
        y_gauss = self._gaussian_peak(padded_w, padded_h)
        self._yf = np.fft.fft2(y_gauss)

        # Train
        xf = np.fft.fft2(patch, axes=(0,1))
        kf = self._gaussian_correlation(xf, xf)
        self._alphaf = self._yf / (kf + self.lambdar)
        self._xf = xf
        
        return True

    def update(self, image):
        if self._roi is None: return False, (0,0,0,0)

        x_center, y_center = self._roi[0] + self._roi[2] // 2, self._roi[1] + self._roi[3] // 2

        x_low_lim, x_high_lim = image.shape[1]*self.dead_zone, image.shape[1] - image.shape[1]*self.dead_zone
        y_low_lim, y_high_lim = image.shape[0]*self.dead_zone, image.shape[0] - image.shape[0]*self.dead_zone
        if (x_low_lim > x_center or x_high_lim < x_center) or (y_low_lim > y_center or y_high_lim < y_center):
            return False, (0, 0, 0, 0)

        # 1. KCF Detection Step (Fast Texture Match)
        patch = self._get_subwindow(image, self._roi, self._template_size)
        zf = np.fft.fft2(patch, axes=(0,1))
        
        kf = self._gaussian_correlation(self._xf, zf)
        response_f = self._alphaf * kf
        response = np.fft.ifft2(response_f).real
        
        # Shift to center
        response_shifted = np.fft.fftshift(response)
        
        # Find KCF Peak
        max_r, max_c = np.unravel_index(np.argmax(response_shifted), response_shifted.shape)
        center_y, center_x = response_shifted.shape[0] // 2, response_shifted.shape[1] // 2
        dy = max_r - center_y
        dx = max_c - center_x
        
        # Proposed new position
        curr_x, curr_y, w, h = self._roi
        prop_x = curr_x + dx
        prop_y = curr_y + dy
        
        # 2. RELIABILITY CHECK (The CSRT-like Logic)
        # Before we accept this new position, let's check if the color makes sense.
        
        # Extract the proposed object region (tight box, no padding)
        img_h, img_w = image.shape[:2]
        px = int(max(0, min(img_w - w, prop_x)))
        py = int(max(0, min(img_h - h, prop_y)))
        
        prop_patch = image[py:py+h, px:px+w]
        
        reliability_score = 0.0
        if prop_patch.size > 0:
            hsv_patch = cv2.cvtColor(prop_patch, cv2.COLOR_BGR2HSV)
            
            # Backproject: For every pixel, how likely is it to be part of our object?
            back_proj = cv2.calcBackProject([hsv_patch], [0, 1], self._hist, [0, 180, 0, 256], 1)
            
            # The score is the average probability of the pixels in the box
            reliability_score = np.mean(back_proj) / 255.0

        # 3. DECISION LOGIC
        
        # A. Total Loss: If color is totally wrong, tracker is lost.
        if reliability_score < self.loss_thresh:
            return False, tuple(self._roi)
            
        # B. Acceptance: The object is valid. Update position.
        self._roi[0] = px
        self._roi[1] = py
        
        # C. Learning Strategy:
        # Only update the KCF model if the view is "Clear" (High reliability).
        # If the object is partially occluded or blurry (Medium reliability),
        # we track it (update pos) but DO NOT update the model. 
        # This prevents learning the occlusion/background.
        
        if reliability_score > self.learning_thresh:
            new_patch = self._get_subwindow(image, self._roi, self._template_size)
            new_xf = np.fft.fft2(new_patch, axes=(0,1))
            kf_new = self._gaussian_correlation(new_xf, new_xf)
            new_alphaf = self._yf / (kf_new + self.lambdar)
            
            self._alphaf = (1 - self.interp_factor) * self._alphaf + self.interp_factor * new_alphaf
            self._xf = (1 - self.interp_factor) * self._xf + self.interp_factor * new_xf
        
        return True, tuple(self._roi)

    def _get_subwindow(self, image, roi, template_size):
        x, y, w, h = map(int, roi)
        cx, cy = x + w // 2, y + h // 2
        tw, th = template_size
        
        patch = cv2.getRectSubPix(image, (tw, th), (cx, cy))
        patch = patch.astype(np.float32) / 255.0
        return patch * self._hann

    def _gaussian_peak(self, w, h):
        output_sigma = np.sqrt(w * h) * self.output_sigma_factor / 0.1
        sy, sx = np.ogrid[-h//2:h//2, -w//2:w//2]
        dist = sx**2 + sy**2
        response = np.exp(-0.5 * (dist / output_sigma**2))
        return np.fft.ifftshift(response)

    def _gaussian_correlation(self, xf, yf):
        N = xf.shape[0] * xf.shape[1]
        xyf = xf * np.conj(yf)
        xy_sum = np.sum(xyf, axis=2)
        xy = np.real(np.fft.ifft2(xy_sum))
        
        xf_sq = np.real(np.fft.ifft2(np.sum(xf * np.conj(xf), axis=2)))
        yf_sq = np.real(np.fft.ifft2(np.sum(yf * np.conj(yf), axis=2)))
        
        xx = xf_sq[0,0]
        yy = yf_sq[0,0]
        
        term = xx + yy - 2 * xy
        term = np.maximum(term, 0) / N
        k = np.exp((-1 / (self.sigma**2)) * term)
        return np.fft.fft2(k)

def TrackerReliabilityKCF_create():
    return ReliabilityKCFTracker()