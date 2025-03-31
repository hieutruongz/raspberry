import cv2
import numpy as np
import socket
import time
from cv2 import aruco

# UDP Communication Setup
UDP_IP = "192.168.4.2"  # Laptop's IP address
UDP_PORT = 9999         # Port for sending data
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Start Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# ArUco Dictionary
arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
arucoParam = aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect ArUco markers
    bboxs, ids, _ = aruco.detectMarkers(frame, arucoDict, parameters=arucoParam)

    if ids is not None:
        center_frame = (frame.shape[1] // 2, frame.shape[0] // 2)  # Frame center

        for i in range(len(ids)):
            # Marker position
            bboxtl = tuple(map(int, bboxs[i][0][0]))  # Top-left corner
            bboxbr = tuple(map(int, bboxs[i][0][2]))  # Bottom-right corner
            center_marker = (
                int((bboxtl[0] + bboxbr[0]) / 2),
                int((bboxtl[1] + bboxbr[1]) / 2)
            )

            # Calculate distance from frame center
            distance = np.linalg.norm(np.array(center_frame) - np.array(center_marker))
            position = (center_marker[0] - center_frame[0], center_marker[1] - center_frame[1])

            # Draw marker, bounding box, and info on frame
            cv2.rectangle(frame, bboxtl, bboxbr, (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {ids[i][0]}", (bboxtl[0], bboxtl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.circle(frame, center_marker, 5, (0, 0, 255), -1)

            # Send ArUco marker data over UDP
            message = f"{ids[i][0]},{position[0]},{position[1]},{distance:.2f}"
            sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
            print(f"Sent: {message}")

    # Display the frame
    cv2.imshow("Aruco Marker Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
sock.close()
