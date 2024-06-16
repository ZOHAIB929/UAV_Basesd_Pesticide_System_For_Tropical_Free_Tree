import cv2
import numpy as np
import torch
model = torch.hub.load('yolov5', 'custom', path='tree.pt', source='local', )  # local repo

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = model(frame, size=640)
    df = result.pandas().xyxy[0]

    # Filter out non-tree detections
    tree_df = df[df['name'] == 'Deciduous Tree']

    # Count tree bounding boxes
    tree_count = len(tree_df)

    for ind in tree_df.index:
        x1, y1 = int(tree_df['xmin'][ind]), int(tree_df['ymin'][ind])
        x2, y2 = int(tree_df['xmax'][ind]), int(tree_df['ymax'][ind])
        label = tree_df['name'][ind]
        conf = tree_df['confidence'][ind]
        text = label + ' ' + str(conf.round(decimals=2))

        # Extract tree area from the frame
        tree_area = frame[y1:y2, x1:x2]

        # Apply color analysis on the tree area
        avg_color_per_row = np.average(tree_area, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        b, g, r = avg_color.astype(np.uint8)

        # Draw bounding box and print color inside the tree
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        radius = int((x2 - x1) / 2)
        cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
        cv2.putText(frame, "Tree", (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(frame, f'Color: ({r}, {g}, {b})', (x1, y1 - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Print tree count
    print('Number of trees:', tree_count)

    # Display the frame with bounding boxes
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()