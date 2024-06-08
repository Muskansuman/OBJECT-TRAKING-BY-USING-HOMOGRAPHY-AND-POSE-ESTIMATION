import cv2 as cv
import numpy as np

# Read the query image
img = cv.imread("img2.jpg", cv.IMREAD_GRAYSCALE)  # Read the image in grayscale
height, width = img.shape[:2]  # Get the height and width of the image

# Display the query image
# cv.imshow("Query Image", img)  # Display the query image (commented out)

# Resize the image
target_width = 800  # Define the target width for the display window
target_height = 600  # Define the target height for the display window

# Calculate the scaling factor for width and height
scale_width = target_width / width
scale_height = target_height / height

# Choose the smallest scaling factor to ensure that the entire image fits within the display window
scale_factor = min(scale_width, scale_height)

# Calculate the new width and height after resizing
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)
img_resized = cv.resize(img, (new_width, new_height))  # Resize the image

# Initialize the video capture
cap = cv.VideoCapture(0)  # Open the default camera

# Initialize the SIFT detector
sift = cv.SIFT_create()  # Create a SIFT object

# Compute keypoints and descriptors for the query image
kp_image, desc_image = sift.detectAndCompute(img_resized, None)  # Detect keypoints and compute descriptors

# Draw keypoints on the query image
img_resized_with_keypoints = cv.drawKeypoints(img_resized, kp_image, img_resized)  # Draw keypoints on the resized image

# Initialize FLANN-based matcher
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv.FlannBasedMatcher(index_params, search_params)  # Create a FLANN-based matcher

while True:
    # Capture frame-by-frame
    _, frame = cap.read()  # Read a new frame from the camera
    grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert frame to grayscale

    # Compute keypoints and descriptors for the train image
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)  # Detect keypoints and compute descriptors for the frame

    # Match keypoints between query and train images
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)  # Match keypoints using FLANN matcher

    # Filter good matches using the Lowe's ratio test
    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)

    # Draw matches on the train image
    img3 = cv.drawMatches(img_resized_with_keypoints, kp_image, grayframe, kp_grayframe, good_points, grayframe,
                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # Draw matches on the frame

    cv.imshow("Matches", img3)  # Display the image with matches

    # If enough good matches are found, estimate homography
    if len(good_points) > 10:
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)  # Find homography matrix
        matches_mask = mask.ravel().tolist()

        try:
            h, w = img_resized.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, matrix)  # Apply perspective transformation
            homography = cv.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)  # Draw the perspective-transformed region
            cv.imshow("Homography", homography)  # Display the homography
            print("Transformation successful.")
        except cv.error as e:
            print("OpenCV error:", e)
            cv.imshow("Homography", grayframe)  # Display the frame if an error occurs

    key = cv.waitKey(1)
    if key == 27:  # If 'Esc' key is pressed, break the loop
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
