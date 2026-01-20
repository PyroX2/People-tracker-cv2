import numpy as np
import cv2


class KCFTracker:
    def __init__(self):
        self.lambdar = 0.0001
        self.sigma = 0.5
        self.output_sigma_factor = 0.125
        self.interp_factor = 0.02   
        self.padding = 1.5
        self.cell_size = 1
        self.dead_zone = 0.07
        
        self.learning_thresh = 0.4
        self.loss_thresh = 0.15
        
        self._roi = None
        self._alphaf = None
        self._xf = None
        self._hann = None
        self._template_size = [0, 0]
        
        self._hist = None

    def init(self, image, bbox):
        x, y, w, h = map(int, bbox)
        
        padded_w = int(w * (1 + self.padding))
        padded_h = int(h * (1 + self.padding))
        
        self._roi = [x, y, w, h]
        self._template_size = [padded_w, padded_h]

        roi_patch = image[y:y+h, x:x+w]
        if roi_patch.size == 0: return False
        
        hsv_roi = cv2.cvtColor(roi_patch, cv2.COLOR_BGR2HSV)

        self._hist = self._calc_hist(hsv_roi, bins=[16, 16])

        hann_rows = np.hanning(padded_h)
        hann_cols = np.hanning(padded_w)
        self._hann = np.outer(hann_rows, hann_cols)[..., None].astype(np.float32)

        patch = self._get_subwindow(image, self._roi, self._template_size)
        
        y_gauss = self._gaussian_peak(padded_w, padded_h)
        self._yf = np.fft.fft2(y_gauss)

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
        
        response_shifted = np.fft.fftshift(response)
        
        max_r, max_c = np.unravel_index(np.argmax(response_shifted), response_shifted.shape)
        center_y, center_x = response_shifted.shape[0] // 2, response_shifted.shape[1] // 2
        dy = max_r - center_y
        dx = max_c - center_x
        
        curr_x, curr_y, w, h = self._roi
        prop_x = curr_x + dx
        prop_y = curr_y + dy
        
        # 2. RELIABILITY CHECK
        img_h, img_w = image.shape[:2]
        px = int(max(0, min(img_w - w, prop_x)))
        py = int(max(0, min(img_h - h, prop_y)))
        
        prop_patch = image[py:py+h, px:px+w]
        
        reliability_score = 0.0
        if prop_patch.size > 0:
            hsv_patch = cv2.cvtColor(prop_patch, cv2.COLOR_BGR2HSV)
            
            back_proj = self._backproject(hsv_patch, self._hist)
            
            reliability_score = np.mean(back_proj) / 255.0

        if reliability_score < self.loss_thresh:
            return False, tuple(self._roi)
            
        self._roi[0] = px
        self._roi[1] = py
        
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

        patch = np.zeros((th, tw, 3))

        x1, y1 = cx - tw // 2, cy - th // 2
        x2, y2 = cx + int(tw // 2), cy + int(th // 2)

        left_padding = 0; right_padding = 0; top_padding = 0; bottom_padding = 0
        img_h, img_w, _ = image.shape

        if x2 > img_w: right_padding = x2 - img_w; x2 = img_w
        if x1 < 0: left_padding = -x1; x1 = 0
        if y2 > img_h: bottom_padding = y2 - img_h; y2 = img_h
        if y1 < 0: top_padding = -y1; y1 = 0

        patch[top_padding:(top_padding+y2-y1), left_padding:(left_padding+x2-x1)] = image[y1:y2, x1:x2]
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
    
    def _calc_hist(self, hsv_image, bins=[16, 16], ranges=[0, 180, 0, 256]):
        h_flat = hsv_image[:, :, 0].flatten()
        s_flat = hsv_image[:, :, 1].flatten()
        
        hist, _, _ = np.histogram2d(
            h_flat, s_flat, 
            bins=bins, 
            range=[[ranges[0], ranges[1]], [ranges[2], ranges[3]]]
        )
        
        if hist.max() > 0:
            hist = (hist / hist.max()) * 255.0
            
        return hist

    def _backproject(self, hsv_patch, hist, ranges=[0, 180, 0, 256]):
        h_bins, s_bins = hist.shape
        
        h = hsv_patch[:, :, 0]
        s = hsv_patch[:, :, 1]
    
        h_step = (ranges[1] - ranges[0]) / h_bins
        s_step = (ranges[3] - ranges[2]) / s_bins
        
        h_indices = (h / h_step).astype(np.int64)
        s_indices = (s / s_step).astype(np.int64)
        
        h_indices = np.clip(h_indices, 0, h_bins - 1)
        s_indices = np.clip(s_indices, 0, s_bins - 1)
        
        back_proj = hist[h_indices, s_indices]

        return np.clip(back_proj, 0, 255).astype(np.uint8)

def TrackerKCF_create():
    return KCFTracker()