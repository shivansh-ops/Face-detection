import cv2

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize video capture from the default webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Read each frame from the video feed
    frame_captured, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optional: Apply histogram equalization to improve contrast
    gray_frame = cv2.equalizeHist(gray_frame)

    # Detect faces with adjusted parameters for accuracy
    detected_faces = face_cascade.detectMultiScale(
        gray_frame, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )

    # Draw rectangles around detected faces
    for (x, y, width, height) in detected_faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
