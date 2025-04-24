import numpy as np
import cv2

class SIFT:
    def __init__(self, sigma=1.6, s=3, num_octaves=4):
        """
         Initialize SIFT parameters.
         
         Parameters:
         - sigma (float): Base sigma for Gaussian blur (default: 1.6)
         - s (int): Number of intervals & differences of gaussian per octave (default: 3)
         - num_octaves (int): Number of octaves (default: 4)
         """
        self.sigma = sigma
        self.s = s # Number of intervals & differences of gaussian per octave
        self.k = 2 ** (1.0 / s) # Scale factor between levels
        self.num_levels = s + 2 # Number of blur levels per octave (s + 2 for DoG)
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

        for i in range(self.num_octaves):
            
            # if 1st octave:
            # base_image for current octave is the original inputted image
            if i == 0:
                base_image = image

            # otherwise, base_image for current octave is:
            else:
                base_image = self.scale_space[i-1][self.s][::2, ::2] # previous octave (i-1), last level (s) in it, downsample the that image by taking every second pixel in both dimensions (reducing the resolution by half).

            octave = []

            # produce the different blurring levels (images) of the current octave
            for level in range(self.num_levels):

                sigma_of_this_level = self.sigma * (self.k ** level) # the Gaussian blur's standard deviation for this level

                gaussian_kernel_size = int(np.ceil(sigma_of_this_level * 3) * 2 + 1)

                if gaussian_kernel_size % 2 == 0:
                    gaussian_kernel_size += 1

                '''
                This line generates a blurred version of the base_image at a specific scale (sigma_of_this_level) using a Gaussian kernel of size gaussian_kernel_size. The sigmaX and sigmaY parameters define the standard deviation of the Gaussian blur in the X and Y directions and ensure isotropic blurring (equal blur in all directions).
                '''
                blurred_image = cv2.GaussianBlur(base_image, (gaussian_kernel_size, gaussian_kernel_size), sigmaX = sigma_of_this_level, sigmaY = sigma_of_this_level)

                octave.append(blurred_image) # add the produced blurred image at current level to the current octave

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

        # 1. get difference of gaussian
        for octave_index in range(self.num_octaves):

            octave_DoG = []
            
            for level in range(self.num_levels - 1): # compute the DoG for all levels except the last one
                dog = self.scale_space[octave_index][level + 1] - self.scale_space[octave_index][level]
                octave_DoG.append(dog) # append dog between current level and the next level

            DoG.append(octave_DoG) # append the full DoG for the current octave

        extrema = []

        # 2. get extrema: compare each DoG pixel with its 26 neighbors
        for octave_index in range(self.num_octaves):

            for level in range(1, self.num_levels - 2): # The loop starts from 1 and ends at self.num_levels - 2 because extrema detection requires comparing each pixel in a DoG image with its neighbors in the previous and next DoG levels

                dog_prev = DoG[octave_index][level - 1]
                dog_curr = DoG[octave_index][level]
                dog_next = DoG[octave_index][level + 1]

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
                            extrema.append((octave_index, level, x, y))
                            
        return extrema

    def localize_keypoints(self, extrema, contrast_threshold=0.03, edge_threshold=10):
        """
         Refine extrema into keypoints with sub-pixel accuracy and filter out unstable ones.
         
         Parameters:
         - extrema (list): List of (octave, level, x, y) from detect_extrema
         - contrast_threshold (float): Minimum DoG magnitude (default: 0.03)
         - edge_threshold (float): 
                - is a parameter used to filter out unstable keypoints that lie on edges.
                - It helps eliminate keypoints with high edge responses by analyzing the curvature of the Difference of Gaussians (DoG) using the Hessian matrix. 
                - If the ratio of principal curvatures (trace²/det) exceeds the edge_threshold, 
                - the keypoint is discarded as it is likely to be on an edge rather than a corner.
         
         Returns:
         - keypoints (list): List of (x, y, sigma) in ORIGINAL IMAGE coordinates
         """
        keypoints = []
        DoG_pyramid = [] # Precompute DoG for all octaves
        for o in range(self.num_octaves):
            octave_DoG = []
            for m in range(self.num_levels - 1):
                dog = self.scale_space[o][m + 1] - self.scale_space[o][m]
                octave_DoG.append(dog)
            DoG_pyramid.append(octave_DoG)

        for (octave, level, x, y) in extrema:

            # Get 3x3x3 DoG neighborhood
            dog_prev = DoG_pyramid[octave][level - 1]
            dog_curr = DoG_pyramid[octave][level]
            dog_next = DoG_pyramid[octave][level + 1]

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
            m_refined = level + offset[2]

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
            scale_factor = 2 ** octave
            x_final = x_refined * scale_factor
            y_final = y_refined * scale_factor
            sigma_final = self.sigma * (self.k ** m_refined) * scale_factor

            kp = cv2.KeyPoint(x_final, y_final, sigma_final * 2)
            keypoints.append(kp)
        return keypoints

    def assign_orientations(self, keypoints, image):

        oriented_keypoints = []

        for key_point in keypoints:

            # x & y position of current key point
            x, y = int(key_point.pt[0]), int(key_point.pt[1])

            sigma = key_point.size / 2

            radius = int(np.ceil(1.5 * sigma))

            if (x - radius < 0 or x + radius >= image.shape[1] or 
                y - radius < 0 or y + radius >= image.shape[0]):
                continue

            patch = image[y-radius : y+radius+1, x-radius : x+radius+1] # from the original image

            dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)

            magnitude = np.sqrt(dx**2 + dy**2)

            # the orientation / direction
            direction = np.arctan2(dy, dx) * 180 / np.pi

            # discretizing the orientations into 36 steps / levels
            direction = (direction + 360) % 360

            y_coords, x_coords = np.indices(patch.shape)

            center = radius

            gaussian = np.exp(-((x_coords - center)**2 + (y_coords - center)**2) / (2 * sigma**2))

            weights = magnitude * gaussian

            hist = np.zeros(36) # hist: histogram of discretized BINS / directions
            
            for i in range(patch.shape[0]):
                for j in range(patch.shape[1]):
                    bin_idx = int(direction[i, j] // 10) # bin: discretized orientation / direction
                    hist[bin_idx] += weights[i, j]

            hist_smoothed = np.convolve(hist, [1, 1, 1], mode='same') / 3

            max_idx = np.argmax(hist_smoothed)

            if hist_smoothed[max_idx] == 0:
                continue
            
            angle = float((max_idx * 10 + 5) % 360)
            new_kp = cv2.KeyPoint(float(x), float(y), key_point.size, angle)
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

    def match_features_with_ncc(self, descriptors1, descriptors2, threshold=0.8):
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


    def match_features_with_ssd(self, descriptors1, descriptors2, threshold = 0.8):
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
        # print (f"matches found: {matches}")
        return matches
    
    def match_and_visualize(self, image1, image2, keypoints1, descriptors1, keypoints2, descriptors2, method='ssd', threshold=0.8, num_matches=150):
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
        # Find matches
        if method == 'ncc':
            matches = self.match_features_with_ncc(descriptors1, descriptors2, threshold=threshold)
        elif method == 'ssd':
            matches = self.match_features_with_ssd(descriptors1, descriptors2, threshold=threshold)

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
        for match in matches[:min(num_matches, len(matches))]: 
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


