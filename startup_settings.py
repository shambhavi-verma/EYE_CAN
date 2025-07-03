import cv2
import numpy as np
import time
import pyautogui as pa

from get_pupil_center import predict_center, init

pa.FAILSAFE = False

model = init()
cap = cv2.VideoCapture(0)

def cal_mean(coordinates_list):
    if not coordinates_list:
        return None

    x_coords = [coord[0] for coord in coordinates_list if coord is not None]
    y_coords = [coord[1] for coord in coordinates_list if coord is not None]

    if len(x_coords) == 0 or len(y_coords) == 0:
        return None

    mean_x = sum(x_coords) / len(x_coords)
    mean_y = sum(y_coords) / len(y_coords)

    return (mean_x, mean_y)

def get_value(cap, number_of_times=20):
    values = []
    for i in range(number_of_times):
        ret, frame = cap.read()
        if not ret:
            break

        nose_coord = process_face(frame)
        if nose_coord is not None:
            values.append(nose_coord)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    mean_value = cal_mean(values)

    return mean_value

def startup(width, height):
    setup_screen = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.putText(setup_screen, "starting setup...", (width//2 - 300, height//2 - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow("setup", setup_screen)   
    cv2.waitKey(2000)


    setup_screen = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(setup_screen, "follow the circles", (width//2 - 300, height//2 - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow("setup", setup_screen)    
    cv2.waitKey(2000)
    cv2.putText(setup_screen, "follow the circles", (width//2 - 300, height//2 - 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    radius = 10

    ## left
    cv2.circle(setup_screen, (radius + 10, height//2), radius, (255,255,255), -1)
    cv2.imshow("setup", setup_screen)    
    cv2.waitKey(2000)
    left = get_value(cap)
    cv2.circle(setup_screen, (radius + 10, height//2), radius, (0,0,0), -1)
    
    ## right
    cv2.circle(setup_screen, (width - radius - 10, height//2), radius, (255,255,255), -1)
    cv2.imshow("setup", setup_screen)    
    cv2.waitKey(2000)
    right = get_value(cap)
    cv2.circle(setup_screen, (width - radius - 10, height//2), radius, (0,0,0), -1)


    ## up
    cv2.circle(setup_screen, (width//2 , radius + 10), radius, (255,255,255), -1)
    cv2.imshow("setup", setup_screen)    
    cv2.waitKey(2000)
    up = get_value(cap)
    cv2.circle(setup_screen, (width//2 , radius + 10), radius, (0,0,0), -1)


    ## down
    cv2.circle(setup_screen, (width//2, height - radius - 10), radius, (255,255,255), -1)
    cv2.imshow("setup", setup_screen)    
    cv2.waitKey(2000)
    down = get_value(cap)
    cv2.circle(setup_screen, (width//2, height - radius - 10), radius, (0,0,0), -1)

    cv2.putText(setup_screen, "setup Complete!", (width//2 - 300, height//2 - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow("setup", setup_screen)    
    cv2.waitKey(2000)
    cv2.destroyWindow("setup")

    return [int(left[0]), int(left[0])], [int(right[0]),int(right[1])], [int(up[0]), int(up[1])], [int(down[0]), int(down[1])]

def process_face(frame):
    return predict_center(model, frame)

def calculate_centroid(coordinates):
    if not coordinates:
        raise ValueError("The list of coordinates is empty.")
    
    # Separate the x and y coordinates
    x_coords, y_coords = zip(*coordinates)
    
    # Calculate the mean of x and y coordinates
    mean_x = sum(x_coords) / len(x_coords)
    mean_y = sum(y_coords) / len(y_coords)
    return [int(mean_x), int(mean_y)]

def smooth_move(x_start, y_start, x_end, y_end, steps, duration):
    x_step = (x_end - x_start) / steps
    y_step = (y_end - y_start) / steps
    for i in range(steps):
        pa.moveTo(x_start + x_step * i, y_start + y_step * i)
        time.sleep(duration / steps)

def map_value(value, from_low, from_high, to_low, to_high):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low
def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

if __name__=="__main__":

    left, right, up, down = startup(1920, 1080)

    print("\n\n",left, right, up, down,"\n\n")

    cap = cv2.VideoCapture(0)

    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)

    mean_coord = []
    per_iteration_mean = 2

    while True:
        ret, frame = cap.read()
        original = frame
        if not ret:
            break

        height, width, _ = frame.shape

        nose_coord = process_face(frame)
        print(nose_coord)
        if nose_coord:
            x_coord = int(map_value(nose_coord[0], left[0], right[0], 10, 1900))
            y_coord = int(map_value(nose_coord[1], up[1], down[1], 10, 1050))
            
            if len(mean_coord)<=per_iteration_mean:
                mean_coord.append([x_coord,y_coord])
            
            else:
                mean_coord = []
                mean_coord.append([x_coord, y_coord])
            if len(mean_coord)==per_iteration_mean:
                position = calculate_centroid(mean_coord)
                pa.moveTo(position[0],position[1], duration=0.2, tween=pa.easeInOutQuad)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

