import mediapipe as mp
import cv2
from function import *

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_mesh_connections = mp.solutions.face_mesh_connections


mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)


def main():

    cap = cv2.VideoCapture(0)

    frame_size_X = 1920
    frame_size_Y = 1080
    old_hand_coords = None
    points_size = 10
    current_color = (255, 0, 0)


    list_of_points = get_list_of_point(frame_size_X, frame_size_Y, 20)

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame,1)
            frame = cv2.resize(frame, (1920, 1080))
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = holistic.process(image)

            # print(dir(results))
            
            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            #& 1. Draw face landmarks
            # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
            #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            #                          )
            
            #& 2. Right hand // Left if Camera is flip
            # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            #                          mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            #                          mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            #                           )

            #& 3. Left Hand // Rigth if Camera is flip
            # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            #                          mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            #                          mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            #                          )

            #& 4. Pose Detections
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
            #                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            #                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            #                          )



            for point in list_of_points:
                draw_point(image, point, int(points_size), current_color)


            if results.left_hand_landmarks:
                h, w, _ = frame.shape
                hand_coords = ([int(results.left_hand_landmarks.landmark[9].x * frame_size_X), int(results.left_hand_landmarks.landmark[9].y * frame_size_Y)])


                if old_hand_coords != None and hand_coords[0] > 0 and hand_coords[0] < 1920 and hand_coords[1] > 0 and hand_coords[1] < 1080:
                 # Define the known points for interpolation
                    x1, y1 = 0, 5
                    x2, y2 = 1920, 100
        
                    # Calculate the linear interpolation
                    points_size = (hand_coords[0] - x1) * (y2 - y1) / (x2 - x1) + y1
        
                    old_hand_coords = hand_coords
                else:
                    old_hand_coords = hand_coords
            

            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
