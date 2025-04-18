import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

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
        self.k = 2 ** (1.0 / s) # Scale factor between levels
        self.num_levels = s + 3 # Number of blur levels per octave (s + 3 for DoG)
        self.num_octaves = num_octaves
        self.scale_space = None # To store the Gaussian scale space

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
        DoG_pyramid = [] # Precompute DoG for all octaves
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
                continue # Skip if Hessian is singular

            # Check if offset is too large (unstable)
            if np.any(np.abs(offset) > 0.5):
                continue # Could iterate here, but we'll skip for simplicity

            # Refined position
            x_refined = x + offset[0]
            y_refined = y + offset[1]
            m_refined = m + offset[2]

            # Compute refined DoG value for contrast check
            D_refined = dog_curr[y, x] + 0.5 * gradient.dot(offset)
            if abs(D_refined) < contrast_threshold:
                continue # Low contrast, discard

            # Edge response elimination (2D Hessian at current level)
            H_2d = np.array([[Dxx, Dxy], [Dxy, Dyy]])
            trace = Dxx + Dyy
            det = Dxx * Dyy - Dxy ** 2
            if det <= 0 or trace ** 2 / det >= (edge_threshold + 1) ** 2 / edge_threshold:
                continue # Edge-like, discard

            # Convert to original image coordinates and sigma
            scale_factor = 2 ** o
            x_final = x_refined * scale_factor
            y_final = y_refined * scale_factor
            sigma_final = self.sigma * (self.k ** m_refined) * scale_factor

            kp = cv2.KeyPoint(x_final, y_final, sigma_final * 2)
            keypoints.append(kp)
        return keypoints

    def assign_orientations(self, keypoints, image):
        oriented_keypoints = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            sigma = kp.size / 2
            radius = int(np.ceil(1.5 * sigma))
            if (x - radius < 0 or x + radius >= image.shape[1] or 
                y - radius < 0 or y + radius >= image.shape[0]):
                continue

            patch = image[y-radius:y+radius+1, x-radius:x+radius+1]
            dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(dx**2 + dy**2)
            direction = np.arctan2(dy, dx) * 180 / np.pi
            direction = (direction + 360) % 360

            y_coords, x_coords = np.indices(patch.shape)
            center = radius
            gaussian = np.exp(-((x_coords - center)**2 + (y_coords - center)**2) / (2 * sigma**2))
            weights = magnitude * gaussian

            hist = np.zeros(36)
            for i in range(patch.shape[0]):
                for j in range(patch.shape[1]):
                    bin_idx = int(direction[i, j] // 10)
                    hist[bin_idx] += weights[i, j]

            hist_smoothed = np.convolve(hist, [1, 1, 1], mode='same') / 3
            max_idx = np.argmax(hist_smoothed)
            if hist_smoothed[max_idx] == 0:
                continue
            
            angle = float((max_idx * 10 + 5) % 360)
            new_kp = cv2.KeyPoint(float(x), float(y), kp.size, angle)
            oriented_keypoints.append(new_kp)
        return oriented_keypoints

    def compute_descriptors(self, keypoints, image):
        filtered_keypoints = []
        descriptors = []
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            sigma = kp.size / 2
            orientation = kp.angle
            radius = int(np.ceil(3 * sigma))
            if (x - radius < 0 or x + radius >= image.shape[1] or 
                y - radius < 0 or y + radius >= image.shape[0]):
                continue

            patch = image[y-radius:y+radius+1, x-radius:x+radius+1]
            dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(dx**2 + dy**2)
            direction = np.arctan2(dy, dx) * 180 / np.pi
            direction = (direction + 360) % 360

            y_coords, x_coords = np.indices(patch.shape)
            center = radius
            gaussian = np.exp(-((x_coords - center)**2 + (y_coords - center)**2) / (2 * (1.5 * sigma)**2))
            weights = magnitude * gaussian

            direction = (direction - orientation + 360) % 360
            patch_size = patch.shape[0]
            subregion_size = patch_size // 4
            descriptor = []

            for i in range(4):
                for j in range(4):
                    y_start = i * subregion_size
                    y_end = (i + 1) * subregion_size
                    x_start = j * subregion_size
                    x_end = (j + 1) * subregion_size
                    sub_weights = weights[y_start:y_end, x_start:x_end]
                    sub_directions = direction[y_start:y_end, x_start:x_end]
                    hist = np.zeros(8)
                    for sy in range(sub_weights.shape[0]):
                        for sx in range(sub_weights.shape[1]):
                            bin_idx = int(sub_directions[sy, sx] // 45)
                            hist[bin_idx] += sub_weights[sy, sx]
                    descriptor.extend(hist)

            descriptor = np.array(descriptor)
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm
                descriptor = np.clip(descriptor, 0, 0.2)
                norm = np.linalg.norm(descriptor)
                if norm > 0:
                    descriptor = descriptor / norm
                    filtered_keypoints.append(kp)
                    descriptors.append(descriptor)

        return filtered_keypoints, np.array(descriptors)

    def compute_ncc(self, descriptor1, descriptor2):
        """
        Compute Normalized Cross-Correlation between two descriptors.
        Args:
            descriptor1: First SIFT descriptor (1D array).
            descriptor2: Second SIFT descriptor (1D array).
        Returns:
            ncc: Normalized Cross-Correlation score (float).
        """
        # Ensure descriptors are float arrays
        d1 = descriptor1.astype(np.float32)
        d2 = descriptor2.astype(np.float32)
        
        # Subtract mean to center the descriptors
        d1 -= np.mean(d1)
        d2 -= np.mean(d2)
        
        # Compute dot product
        numerator = np.dot(d1, d2)
        
        # Compute norms (standard deviations)
        norm1 = np.sqrt(np.sum(d1 ** 2))
        norm2 = np.sqrt(np.sum(d2 ** 2))
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute NCC
        ncc = numerator / (norm1 * norm2)
        return ncc

    def match_features(self, descriptors1, descriptors2, threshold=0.8):
        """
        Match SIFT descriptors between two images using NCC, minimizing distance.
        Args:
            descriptors1: SIFT descriptors of first image (n1 x 128).
            descriptors2: SIFT descriptors of second image (n2 x 128).
            threshold: Minimum NCC score to consider a match (default: 0.8).
        Returns:
            matches: List of cv2.DMatch objects representing best matches.
        """
        matches = []
        
        # Iterate over each descriptor in the first image
        for i in range(len(descriptors1)):
            best_distance = float('inf')
            best_idx = -1
            
            # Compare with each descriptor in the second image
            for j in range(len(descriptors2)):
                ncc = self.compute_ncc(descriptors1[i], descriptors2[j])
                distance = 1.0 - ncc  # Convert NCC to distance
                
                # Update best match if distance is lower
                if distance < best_distance:
                    best_distance = distance
                    best_idx = j
            
            # Store match if NCC exceeds the threshold (i.e., distance is low enough)
            if (1.0 - best_distance) >= threshold:
                match = cv2.DMatch()
                match.queryIdx = i
                match.trainIdx = best_idx
                match.distance = best_distance
                matches.append(match)
        
        # Sort matches by distance (ascending)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches


    def match_featues_with_ssd(self, descriptors1, descriptors2, threshold = 0.8):
        """
        Match SIFT descriptors between two images using SSD, minimizing distance.
        Args:
            descriptors1: SIFT descriptors of first image (n1 x 128).
            descriptors2: SIFT descriptors of second image (n2 x 128).
        Returns:
            matches: List of cv2.DMatch objects representing best matches.
        """
        matches = []
        
        for i in range(len(descriptors1)):
            best_distance = float('inf')
            best_idx = -1
            
            for j in range(len(descriptors2)):
                distance = np.sum((descriptors1[i] - descriptors2[j]) ** 2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_idx = j
            
            if best_distance <= 1- threshold:
                # Store match
                match = cv2.DMatch()
                match.queryIdx = i
                match.trainIdx = best_idx
                match.distance = best_distance
                matches.append(match)
        
        # Sort matches by distance (ascending)
        matches = sorted(matches, key=lambda x: x.distance)
        print (f"matches found: {matches}")
        return matches
    
    
    # def match_and_visualize(self, image1, image2, keypoints1, descriptors1, keypoints2, descriptors2, sift):
    #     """
    #     Match features between two images and visualize the results.
    #     Args:
    #         image1: First grayscale image (numpy array).
    #         image2: Second grayscale image (numpy array).
    #         keypoints1: List of cv2.KeyPoint objects for first image.
    #         descriptors1: SIFT descriptors for first image (n1 x 128).
    #         keypoints2: List of cv2.KeyPoint objects for second image.
    #         descriptors2: SIFT descriptors for second image (n2 x 128).
    #         sift: SIFT object with match_features method.
    #     Returns:
    #         matched_image: Image with drawn matches (BGR).
    #     """
    #     # Find matches using NCC
    #     # matches = sift.match_features(descriptors1, descriptors2, threshold=0.8)
    #     matches = sift.match_featues_with_ssd(descriptors1, descriptors2)
    #     # Convert grayscale images to BGR for color visualization
    #     img1_color = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    #     img2_color = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        
    #     # Draw top N matches (e.g., top 50 or all if fewer)
    #     matched_image = cv2.drawMatches(
    #         img1_color, keypoints1, img2_color, keypoints2, matches[:min(50, len(matches))],
    #         None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    #         matchColor=(0, 255, 0),  # Green lines for matches
    #         singlePointColor=(0, 0, 255)  # Red for unmatched keypoints
    #     )
        
    #     print(f"Number of matches found: {len(matches)}")
    #     return matched_image

    def match_and_visualize(self, image1, image2, keypoints1, descriptors1, keypoints2, descriptors2, sift):
        """
        Match features between two images and visualize the results.
        Args:
            image1: First grayscale image (numpy array).
            image2: Second grayscale image (numpy array).
            keypoints1: List of cv2.KeyPoint objects for first image.
            descriptors1: SIFT descriptors for first image (n1 x 128).
            keypoints2: List of cv2.KeyPoint objects for second image.
            descriptors2: SIFT descriptors for second image (n2 x 128).
            sift: SIFT object with match_features method.
        Returns:
            matched_image: Image with drawn matches (BGR).
        """
        # Find matches using SSD
        matches = sift.match_featues_with_ssd(descriptors1, descriptors2)

        # Convert grayscale images to BGR for color visualization
        img1_color = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

        # Create a canvas to combine both images side by side
        height1, width1 = img1_color.shape[:2]
        height2, width2 = img2_color.shape[:2]
        combined_height = max(height1, height2)
        combined_width = width1 + width2
        matched_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        matched_image[:height1, :width1] = img1_color
        matched_image[:height2, width1:] = img2_color

        # Draw each match with a unique color
        for match in matches[:min(50, len(matches))]:  # Limit to top 50 matches
            # Generate a random color
            color = tuple(np.random.randint(0, 256, 3).tolist())

            # Get the keypoints for the match
            pt1 = tuple(map(int, keypoints1[match.queryIdx].pt))
            pt2 = tuple(map(int, keypoints2[match.trainIdx].pt))
            pt2 = (pt2[0] + width1, pt2[1])  # Adjust pt2's x-coordinate for the combined image

            # Draw a line between the matched points
            cv2.line(matched_image, pt1, pt2, color, 2)

            # Draw circles at the keypoints
            cv2.circle(matched_image, pt1, 5, color, -1)
            cv2.circle(matched_image, pt2, 5, color, -1)

        print(f"Number of matches found: {len(matches)}")
        return matched_image

def select_image():
    app = QApplication(sys.argv)
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getOpenFileName(
        None, 
        "Select Image", 
        "", 
        "Image Files (*.jpg *.jpeg *.png *.bmp *.gif)", 
        options=options
    )
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {file_path}")
            return None
        return img
    return None

if __name__ == "__main__":
    # Select two images
    print("Select the first image:")
    image1 = select_image()
    print("Select the second image:")
    image2 = select_image()
    
    if image1 is None or image2 is None:
        print("One or both images failed to load or no images selected.")
        sys.exit(1)
    
    # Initialize SIFT
    sift = SIFT(sigma=1.6, s=3, num_octaves=4)
    
    # Process first image
    sift.build_scale_space(image1)
    extrema1 = sift.detect_extrema()
    keypoints1 = sift.localize_keypoints(extrema1)
    oriented_keypoints1 = sift.assign_orientations(keypoints1, image1)
    final_keypoints1, descriptors1 = sift.compute_descriptors(oriented_keypoints1, image1)
    
    print(f"Image 1: Found {len(keypoints1)} initial keypoints")
    print(f"Image 1: After orientation assignment: {len(oriented_keypoints1)} keypoints")
    print(f"Image 1: After descriptor computation: {len(final_keypoints1)} keypoints with {descriptors1.shape} descriptors")
    
    # Process second image
    sift.build_scale_space(image2)
    extrema2 = sift.detect_extrema()
    keypoints2 = sift.localize_keypoints(extrema2)
    oriented_keypoints2 = sift.assign_orientations(keypoints2, image2)
    final_keypoints2, descriptors2 = sift.compute_descriptors(oriented_keypoints2, image2)
    
    print(f"Image 2: Found {len(keypoints2)} initial keypoints")
    print(f"Image 2: After orientation assignment: {len(oriented_keypoints2)} keypoints")
    print(f"Image 2: After descriptor computation: {len(final_keypoints2)} keypoints with {descriptors2.shape} descriptors")
    
    # Perform feature matching and visualize
    if len(final_keypoints1) > 0 and len(final_keypoints2) > 0:
        matched_image = sift.match_and_visualize(
            image1, image2, final_keypoints1, descriptors1, final_keypoints2, descriptors2, sift
        )
        
        # Display and save the result
        cv2.imshow("SIFT Feature Matches", matched_image)
        cv2.imwrite("matched_output.jpg", matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No keypoints found in one or both images. Cannot perform matching.")