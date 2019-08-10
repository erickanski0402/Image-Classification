from keras.models import model_from_json
import numpy as np
import cv2

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
print('Loaded model from disk')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(frame, dsize = (64,64), interpolation = cv2.INTER_CUBIC)
    guess = loaded_model.predict(img.reshape(1,64,64,3))

    if guess > 0.5:
        print("DOG")
    else:
        print("CAT")

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
