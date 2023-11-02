import mediapipe as mp
import cv2
import random

def generate_point(frame_size_X, frame_size_Y):
    point = (random.randint(0, frame_size_X - 1), random.randint(0, frame_size_Y - 1))
    return point
    
def draw_point(image, point,  size = 10, color = (255, 0, 0), thickness = cv2.FILLED):
    cv2.circle(image, (point[0], point[1]), size, color, thickness)

def get_list_of_point(frame_size_X, frame_size_Y, point_number):
    points = []
    for i in range (0, point_number):
        points.append(generate_point(frame_size_X, frame_size_Y))

    return points