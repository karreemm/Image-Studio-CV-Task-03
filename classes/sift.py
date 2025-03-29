import numpy as np
import cv2

class SIFT:
    def __init__(self, sigma=1.6, s=3, num_octaves=4):
        """
        Initialize SIFT parameters.
        
        Parameters:
        - sigma (float): Base sigma for Gaussian blur (default: 1.6)
        - s (int): Number of intervals per octave (default: 3)
        - num_octaves (int): Number of octaves (default: 4)
        """
        self.sigma = sigma
        self.s = s
        self.k = 2 ** (1.0 / s)  # Scale factor between levels
        self.num_levels = s + 3   # Number of blur levels per octave (s + 3 for DoG)
        self.num_octaves = num_octaves
        self.scale_space = None   # To store the Gaussian scale space

    def build_scale_space(self, image):
        """
        Construct the Gaussian scale space for the input image using cv2.GaussianBlur.
        
        Parameters:
        - image (ndarray): Grayscale input image as a 2D NumPy array
        """
        image = image.astype(np.float32)
        self.scale_space = []

        for o in range(self.num_octaves):
            if o == 0:
                base_image = image
            else:
                base_image = self.scale_space[o-1][self.s][::2, ::2]

            octave = []
            for m in range(self.num_levels):
                sigma_m = self.sigma * (self.k ** m)
                ksize = int(np.ceil(sigma_m * 3) * 2 + 1)
                if ksize % 2 == 0:
                    ksize += 1
                L = cv2.GaussianBlur(base_image, (ksize, ksize), sigmaX=sigma_m, sigmaY=sigma_m)
                octave.append(L)
            self.scale_space.append(octave)

    def detect_extrema(self):
        """
        Detect scale space extrema in the Difference of Gaussians (DoG).
        
        Returns:
        - extrema (list): List of tuples (octave, level, x, y) representing extrema locations
        """
        if self.scale_space is None:
            raise ValueError("Scale space has not been constructed. Call build_scale_space first.")

        DoG = []
        for o in range(self.num_octaves):
            octave_DoG = []
            for m in range(self.num_levels - 1):
                dog = self.scale_space[o][m + 1] - self.scale_space[o][m]
                octave_DoG.append(dog)
            DoG.append(octave_DoG)

        extrema = []
        for o in range(self.num_octaves):
            for m in range(1, self.num_levels - 2):
                dog_prev = DoG[o][m - 1]
                dog_curr = DoG[o][m]
                dog_next = DoG[o][m + 1]
                height, width = dog_curr.shape

                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        val = dog_curr[y, x]
                        neighbors = [
                            dog_curr[y-1, x-1], dog_curr[y-1, x], dog_curr[y-1, x+1],
                            dog_curr[y, x-1],                     dog_curr[y, x+1],
                            dog_curr[y+1, x-1], dog_curr[y+1, x], dog_curr[y+1, x+1],
                            dog_prev[y-1, x-1], dog_prev[y-1, x], dog_prev[y-1, x+1],
                            dog_prev[y, x-1],   dog_prev[y, x],   dog_prev[y, x+1],
                            dog_prev[y+1, x-1], dog_prev[y+1, x], dog_prev[y+1, x+1],
                            dog_next[y-1, x-1], dog_next[y-1, x], dog_next[y-1, x+1],
                            dog_next[y, x-1],   dog_next[y, x],   dog_next[y, x+1],
                            dog_next[y+1, x-1], dog_next[y+1, x], dog_next[y+1, x+1]
                        ]
                        if val > max(neighbors) or val < min(neighbors):
                            extrema.append((o, m, x, y))

        return extrema

    def localize_keypoints(self, extrema, contrast_threshold=0.03, edge_threshold=10):
        """
        Refine extrema into keypoints with sub-pixel accuracy and filter out unstable ones.
        
        Parameters:
        - extrema (list): List of (octave, level, x, y) from detect_extrema
        - contrast_threshold (float): Minimum DoG magnitude (default: 0.03)
        - edge_threshold (float): Maximum curvature ratio for edge rejection (default: 10)
        
        Returns:
        - keypoints (list): List of (x, y, sigma) in original image coordinates
        """
        keypoints = []
        DoG_pyramid = []  # Precompute DoG for all octaves
        for o in range(self.num_octaves):
            octave_DoG = []
            for m in range(self.num_levels - 1):
                dog = self.scale_space[o][m + 1] - self.scale_space[o][m]
                octave_DoG.append(dog)
            DoG_pyramid.append(octave_DoG)

        for (o, m, x, y) in extrema:
            # Get 3x3x3 DoG neighborhood
            dog_prev = DoG_pyramid[o][m - 1]
            dog_curr = DoG_pyramid[o][m]
            dog_next = DoG_pyramid[o][m + 1]

            # Check bounds (skip if too close to edge)
            if x < 1 or y < 1 or x >= dog_curr.shape[1] - 1 or y >= dog_curr.shape[0] - 1:
                continue

            # Sub-pixel refinement
            # First derivatives
            Dx = (dog_curr[y, x+1] - dog_curr[y, x-1]) / 2.0
            Dy = (dog_curr[y+1, x] - dog_curr[y-1, x]) / 2.0
            Ds = (dog_next[y, x] - dog_prev[y, x]) / 2.0
            gradient = np.array([Dx, Dy, Ds])

            # Second derivatives
            Dxx = dog_curr[y, x+1] - 2 * dog_curr[y, x] + dog_curr[y, x-1]
            Dyy = dog_curr[y+1, x] - 2 * dog_curr[y, x] + dog_curr[y-1, x]
            Dss = dog_next[y, x] - 2 * dog_curr[y, x] + dog_prev[y, x]
            Dxy = (dog_curr[y+1, x+1] - dog_curr[y+1, x-1] - dog_curr[y-1, x+1] + dog_curr[y-1, x-1]) / 4.0
            Dxs = (dog_next[y, x+1] - dog_next[y, x-1] - dog_prev[y, x+1] + dog_prev[y, x-1]) / 4.0
            Dys = (dog_next[y+1, x] - dog_next[y-1, x] - dog_prev[y+1, x] + dog_prev[y-1, x]) / 4.0
            hessian = np.array([
                [Dxx, Dxy, Dxs],
                [Dxy, Dyy, Dys],
                [Dxs, Dys, Dss]
            ])

            # Solve for offset: x̂ = -H⁻¹ * ∇D
            try:
                offset = -np.linalg.inv(hessian).dot(gradient)
            except np.linalg.LinAlgError:
                continue  # Skip if Hessian is singular

            # Check if offset is too large (unstable)
            if np.any(np.abs(offset) > 0.5):
                continue  # Could iterate here, but we'll skip for simplicity

            # Refined position
            x_refined = x + offset[0]
            y_refined = y + offset[1]
            m_refined = m + offset[2]

            # Compute refined DoG value for contrast check
            D_refined = dog_curr[y, x] + 0.5 * gradient.dot(offset)
            if abs(D_refined) < contrast_threshold:
                continue  # Low contrast, discard

            # Edge response elimination (2D Hessian at current level)
            H_2d = np.array([
                [Dxx, Dxy],
                [Dxy, Dyy]
            ])
            trace = Dxx + Dyy
            det = Dxx * Dyy - Dxy ** 2
            if det <= 0 or trace ** 2 / det >= (edge_threshold + 1) ** 2 / edge_threshold:
                continue  # Edge-like, discard

            # Convert to original image coordinates and sigma
            scale_factor = 2 ** o
            x_final = x_refined * scale_factor
            y_final = y_refined * scale_factor
            sigma_final = self.sigma * (self.k ** m_refined) * scale_factor

            keypoints.append((x_final, y_final, sigma_final))

        return keypoints

# Example usage:
if __name__ == "__main__":
    # Create a dummy 64x64 grayscale image
    image = np.random.rand(64, 64) * 255
    sift = SIFT(sigma=1.6, s=3, num_octaves=4)
    sift.build_scale_space(image)
    extrema = sift.detect_extrema()
    keypoints = sift.localize_keypoints(extrema)
    print(f"Found {len(keypoints)} keypoints: {keypoints[:5]}")  # Print first 5