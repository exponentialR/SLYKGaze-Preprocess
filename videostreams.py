
import cv2

cap = cv2.VideoCapture(0)
dim = (640, 480)

while True:
    ret, image = cap.read()
    print(image.shape[:2])
    crop = image[100:200 + 100, 0:100 + 100]
    print(crop.shape)
    resized_image = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
    print(resized_image.shape)
    cv2.imshow('Original Window', image)
    cv2.imshow('Cropped Window', resized_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
