import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from get_pupil_center import predict_center, init
from startup_settings import startup
import math
import time

model = init()

# Start OpenCV video stream
cap = cv2.VideoCapture(0)

flag = 0

# Initialize session state to track which page the user is on
if 'page' not in st.session_state:
    st.session_state.page = 'main'  # Default page is 'main'

# Function to switch pages
def switch_page(new_page):
    st.session_state.page = new_page

def map_value(value, from_low, from_high, to_low, to_high):
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low

def calculate_centroid(coordinates):
    if not coordinates:
        raise ValueError("The list of coordinates is empty.")
    
    # Separate the x and y coordinates
    x_coords, y_coords = zip(*coordinates)
    
    # Calculate the mean of x and y coordinates
    mean_x = sum(x_coords) / len(x_coords)
    mean_y = sum(y_coords) / len(y_coords)
    return [int(mean_x), int(mean_y)]


def find_closest_box(box_centers, location):
    # Initialize variables to store the closest box and the minimum distance
    closest_box = None
    min_distance = float('inf')  # Set initial distance to a very large number
    
    # Iterate over the dictionary to calculate the distance of each box from the location
    for box, center in box_centers.items():
        # Calculate the Euclidean distance between the center of the box and the location
        distance = math.sqrt((center[0] - location[0]) ** 2 + (center[1] - location[1]) ** 2)
        
        # If this distance is smaller than the current minimum, update the closest box
        if distance < min_distance:
            min_distance = distance
            closest_box = box
    
    return box_centers[closest_box]

def main_page():
    global flag
    # Streamlit webapp code
    st.markdown("<h1 style='text-align: center;'>EyeCan</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This project involves a spectacle-like device equipped with a camera for real-time eye movement monitoring using Mediapipe and OpenCV. Eye movements and gestures are detected with Hough Circle and custom CNN models. Communication between the device and applications is handled via Django Channels, and the system is integrated with web and desktop apps using Django and PostgreSQL to enhance job optimization and accessibility. Designed to empower disabled individuals, the device enables control of computer functions through eye movements, promoting independence, accessibility, and career opportunities while minimizing environmental impact.</p>", unsafe_allow_html=True)

    callibrate = st.button('Callibrate')

    if callibrate==True:
        flag = 0
        switch_page('job1')


screen_width = 1900  # Example width of the screen
screen_height = 1060  # Example height of the screen


# Function to calculate the centers of the boxes based on screen size
def calculate_box_centers(screen_width, screen_height):
    # Define the box size
    box_height = 400
    box_width = screen_width / 2  # Each box occupies 50% width of the screen

    # Calculate the centers for each box in the 2x2 grid
    centers = {
        "box1": (int(box_width / 2), int(box_height / 2)),  # Top-left box
        "box2": (int(screen_width - box_width / 2), int(box_height / 2)),  # Top-right box
        "box3": (int(box_width / 2), int(screen_height - box_height / 2)),  # Bottom-left box
        "box4": (int(screen_width - box_width / 2), int(screen_height - box_height / 2))  # Bottom-right box
    }
    return centers

def generate_circle_style(x, y):
    return f"""
    position: absolute;
    top: {y}px;    /* Y-coordinate */
    left: {x}px;   /* X-coordinate */
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background-color: rgba(128, 128, 128, 0.5);  /* Gray color with 50% transparency */
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 16px;
    color: white;
    font-weight: bold;
    """

def closest_coordinate(coord_list, coord):

    # Convert the list and the coordinate to NumPy arrays
    coord_list = np.array(coord_list)
    coord = np.array(coord)
    
    # Calculate the Euclidean distance between the coord and each point in coord_list
    distances = np.linalg.norm(coord_list - coord, axis=1)

    # Find the index of the smallest distance
    closest_index = np.argmin(distances)

    # Return the closest coordinate
    return coord_list[closest_index], closest_index

def job1():
    error_placeholder = st.empty()

    global flag
    if flag==0:
        left, right, up, down = startup(1920, 1080)
        print(left, right, up, down)
        if left==None or right==None or up==None or down==None :
            error_placeholder.text(f"Startup Values are Empty : \nLeft:{left}\nRight:{right}\nUp:{up}\nDown:{down}\n")

        # Initialize a placeholder for displaying the coordinates
        coordinates_placeholder = st.empty()
        position_placeholder = st.empty()

        # Add a stop button (outside the loop to avoid duplication issues)
        stop = st.button('Stop')

        mean_coord = []
        per_iteration_mean = 2

        # Create a blank image (black background)
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)

        blink = False

        # Main loop to capture frames and predict coordinates
        while not stop:
            ret, frame = cap.read()

            if not ret:
                st.write("Failed to grab frame")
                break
            
            cv2.putText(img, "Which one is the correct category ?", (530,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # Convert the OpenCV frame (BGR) to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to PIL format for YOLO
            img_pil = Image.fromarray(frame_rgb)

            # Get the detected center coordinates
            eye_coord = predict_center(model, img_pil)


            if eye_coord:
                coordinates_placeholder.text(f"Center Coordinates: {eye_coord}")
            
            # else:
            if eye_coord==None:
                ## no object is the condition Of CLICKing
                coordinates_placeholder.text("No object detected")
                blink = True
                print(f'Blink : {blink}\n')
            else:
                blink = False
                print(f'Blink : {blink}\n')

            width, height = 1910, 1070

            # if eye_coord!=None:
            try:
                # Mapped coordinates
                x_coord = int(map_value(eye_coord[0], left[0], right[0], 10, 1900))
                y_coord = int(map_value(eye_coord[1], up[1], down[1], 10, 1050))


                if len(mean_coord)<=per_iteration_mean:
                    mean_coord.append([x_coord,y_coord])
                else:
                    mean_coord = []
                    mean_coord.append([x_coord,y_coord])
                
                if len(mean_coord)==per_iteration_mean:
                    position = calculate_centroid(mean_coord)

                    if position:
                        position_placeholder.text(f"Position Coordinates: {position}")
                    else:
                        position_placeholder.text("None")

                    center_A = (477,207) # y - 40
                    center_B = (1431,207) # y - 40
                    center_C = (477, 782)
                    center_D = (1431, 782)

                    # List of centers
                    centers = [center_A, center_B, center_C, center_D]

                    closest, box_index = closest_coordinate(centers, position)

                    box_width = 150
                    box_height = 100
                    ##Draw the boxes and text
                    cv2.putText(img, "Navigate...", (477, 10), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 5)
                    for i, coord in enumerate(centers):
                        # Draw rectangles
                        print(f'\n  blink near color : {blink}\n')
                        if i==box_index:
                            color = (255, 0, 0)
                        else:
                            color = (255, 255, 255)
                        if blink==True:
                            color = (0, 255,0)


                        cv2.rectangle(img, (coord[0]-box_width, coord[1]-box_height), (coord[0]+box_width, coord[1]+box_height), color, -1)
                        # cv2.circle(img, coord, 10, (234,133,45), 2)
                        cv2.putText(img, "A", center_A, cv2.FONT_HERSHEY_SIMPLEX, 1, (30,30,30), 2)
                        cv2.putText(img, "B", center_B, cv2.FONT_HERSHEY_SIMPLEX, 1, (30,30,30), 2)
                        cv2.putText(img, "C", center_C, cv2.FONT_HERSHEY_SIMPLEX, 1, (30,30,30), 2)
                        cv2.putText(img, "D", center_D, cv2.FONT_HERSHEY_SIMPLEX, 1, (30,30,30), 2)

                    # Show the image in the OpenCV window
                    cv2.imshow("Coordinates Display", img)
        
                    ## Check for key press to break the loop
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    # cv2.putText(img, str(position), (width//2,height//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
            except Exception as e:
                print(f'Exception : {e}')
                
            # Check for stop button press to break out of loop
            print(f"\nStop : {stop}\n")
            if stop==True:
                flag = 1
                break


        # Release the camera when done
        cap.release()
        cv2.destroyAllWindows()
    else:
        switch_page('main_page')

if st.session_state.page == 'main':
    main_page()
else:
    job1()