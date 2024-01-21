#from tensorflow.keras.models import load_model
from skimage.transform import resize
import pickle
import numpy as np
import cv2

#TF_ENABLE_ONEDNN_OPTS=0
EMPTY = True
NOT_EMPTY = False

#MODEL = load_model('Classification_for_car\model.h5')
MODEL=pickle.load(open("Classification_for_car\model.p","rb"))


def empty_or_not(spot_bgr):
    flat_data=[]
    img_resized = resize(spot_bgr, (125,125, 3))
    #image_array = np.array(img_resized) / 255.0  # Normalize pixel values to be in the range [0, 1]
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)/255.0

    y_output = MODEL.predict(flat_data)
    # Add an extra dimension to match the input shape expected by the model
    #image_array = np.expand_dims(image_array, axis=0)

    # Predict using the model
    #y_output = MODEL.predict(image_array)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

