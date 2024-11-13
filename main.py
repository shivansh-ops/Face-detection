import cv2

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the video capture from the default webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Read each frame from the video feed
    frame_captured, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    detected_faces = face_cascade.detectMultiScale(gray_frame, 1.3, 6)

    # Draw rectangles around detected faces
    for (x, y, width, height) in detected_faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 5)

    # Display the frame with the detected faces
    cv2.imshow('Face Detection', frame)

    # Wait for a key press; exit the loop if the key 'q' is pressed
    key = cv2.waitKey(40) & 0xff
    if key == ord('q'):
        break

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
