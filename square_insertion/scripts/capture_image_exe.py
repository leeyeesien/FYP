import cv2
import time
import os.path

def capture_image(image_id):
    camera = cv2.VideoCapture(2)
    delta = 0
    previous = time.time()
    while True:
        current  = time.time()
        delta += current - previous
        previous = current

        # Show the image and keep streaming
        _, img = camera.read()
        cv2.imshow("Frame", img)
        cv2.waitKey(1)

        # Check if 1 (or some other value) seconds passed
        if delta > 1:
            # Operations on image
            # Reset the time counter
            cv2.imwrite(os.path.dirname(__file__) + '/../ins_exe_data/img/'+ str(image_id) +'.png', img)
            delta = 0
            camera.release()
            cv2.destroyAllWindows() 
            # break
            return img