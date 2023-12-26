import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "LHip": 10, "LKnee": 11, "Chest": 12}

POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
              ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
              ["Chest", "LHip"], ["LHip", "LKnee"]]

# Load the data
input_data = pd.read_csv('dataset/movement.csv')
output_data = pd.read_csv('dataset/animate.csv')

# Split the data into training and testing sets (70% training, 30% testing)
input_train, input_test, output_train, output_test = train_test_split(input_data, output_data, test_size=0.3,
                                                                      random_state=42)

# Create a Random Forest Regressor with MultiOutputRegressor
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
model = MultiOutputRegressor(random_forest)

# Fit the model
model.fit(input_train, output_train)

# Make predictions on the test set
predictions = model.predict(input_test)

# Create an empty canvas
canvas = 255 * np.ones((480, 640, 3), dtype=np.uint8)

# Scale factor for drawing points on the canvas
scale_factor = 10

# Draw points and lines on the canvas
for pose_pair in POSE_PAIRS:
    part1_index = BODY_PARTS[pose_pair[0]]
    part2_index = BODY_PARTS[pose_pair[1]]

    for i in range(len(predictions)):
        # Extract x, y positions from predictions
        x1, y1 = int(predictions[i][2 * part1_index] * scale_factor), int(
            predictions[i][2 * part1_index + 1] * scale_factor)
        x2, y2 = int(predictions[i][2 * part2_index] * scale_factor), int(
            predictions[i][2 * part2_index + 1] * scale_factor)

        # Draw points
        cv2.circle(canvas, (x1, y1), 5, (0, 0, 255), -1)
        cv2.circle(canvas, (x2, y2), 5, (0, 0, 255), -1)

        # Draw lines
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the canvas
cv2.imshow('Pose Estimation', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
