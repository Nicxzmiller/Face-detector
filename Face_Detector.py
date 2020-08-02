import cv2

# loading data
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in
img = cv2.imread('girl.jpg')

# convert images to greyscale
grayscaled_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_data.detectMultiScale(grayscaled_image)

# draw rectangles around the faces
(x, y, w, h) = face_coordinates[0]
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# print(face_coordinates)

#
cv2.imshow('Nicks Face Detector', img)

# wait until a key is pressed
cv2.waitKey()

print("code completed")