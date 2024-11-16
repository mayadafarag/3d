
import cv2
import numpy as np

def find_feature_matches(img1, img2):
    # Use ORB (Oriented FAST and Rotated BRIEF) detector
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Match descriptors using KNN matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    return points1, points2, matches

def estimate_fundamental_matrix(points1, points2):
    # Estimate the fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    return F, mask

def reconstruct_3D(points1, points2, K):
    # Compute the essential matrix using the intrinsic camera matrix K
    E, _ = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, points1, points2, K)
    
    # Project points into 3D space using triangulation
    proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))         # First camera matrix
    proj2 = np.hstack((R, t))                                # Second camera matrix
    points_4D = cv2.triangulatePoints(K @ proj1, K @ proj2, points1.T, points2.T)
    points_3D = points_4D[:3] / points_4D[3]                 # Convert to homogeneous coordinates
    return points_3D.T

# Load stereo images
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# Example intrinsic camera matrix K (adjust with your camera calibration parameters)
K = np.array([[1000, 0, 320], 
              [0, 1000, 240], 
              [0, 0, 1]])

# Step 1: Find feature matches
points1, points2, matches = find_feature_matches(img1, img2)

# Step 2: Estimate the fundamental matrix
F, mask = estimate_fundamental_matrix(points1, points2)

# Step 3: 3D reconstruction
points_3D = reconstruct_3D(points1[mask.ravel() == 1], points2[mask.ravel() == 1], K)

# Print reconstructed 3D points
print("Reconstructed 3D points:", points_3D)

