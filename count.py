import cv2
from util import parking_spots_bboxes, empty_or_not

mask_path = 'mask2.png'
video_path = 'parking_space2.mp4'

mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = parking_spots_bboxes(connected_components)

spots_status=[None for j in spots]
frame_num=0
ret = True
step=120
while True:
    ret, frame = cap.read()

    if frame_num % step ==0:

        for spot_index,spot in enumerate(spots):
            x1, y1, w, h = spot

            # Check if y1 is equal to 560, 563, or 595 and skip drawing the rectangle
            if y1 in {560, 563, 595}:
                continue

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_index]=spot_status

    for spot_index,spot in enumerate(spots):
        spot_status=spots_status[spot_index]
        x1,y1,w,h=spots[spot_index]

        if spot_status:
              
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
               
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    # Resize the frame
    #frame = cv2.resize(frame, (1280, 800))

    cv2.imshow('frame', frame)

    # Wait for 25 milliseconds and check if the 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_num+=1

cap.release()
cv2.destroyAllWindows()
