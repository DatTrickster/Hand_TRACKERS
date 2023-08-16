import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def calculate_average_hand_position(hand_landmarks):
    x_sum, y_sum, z_sum = 0, 0, 0
    for landmark in hand_landmarks.landmark:
        x_sum += landmark.x
        y_sum += landmark.y
        z_sum += landmark.z

    averaged_x = x_sum / len(hand_landmarks.landmark)
    averaged_y = y_sum / len(hand_landmarks.landmark)
    averaged_z = z_sum / len(hand_landmarks.landmark)

    return averaged_x, averaged_y, averaged_z

def calibrate_hand(cap):
    calib_count = 0
    calib_frames = 50
    x_sum, y_sum, z_sum = 0, 0, 0

    while calib_count < calib_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for the black and white feed
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale frame to RGB for hand tracking feed
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Process the frame to detect hands
        with mp.solutions.hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands:
            results = hands.process(rgb_frame)

            # Check if any hands were detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x, y, z = calculate_average_hand_position(hand_landmarks)
                    x_sum += x
                    y_sum += y
                    z_sum += z

                calib_count += 1

        # Display the frame with the hand tracking
        cv2.imshow("Calibration", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    avg_x = x_sum / calib_frames
    avg_y = y_sum / calib_frames
    avg_z = z_sum / calib_frames

    cv2.destroyAllWindows()
    return avg_x, avg_y, avg_z

def track_hands():
    cap = cv2.VideoCapture(0)

    # Set the frame width and height as desired
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # Increase the width to make tracking space bigger
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)  # Increase the height to make tracking space bigger

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    # Calibrate the hand
    print("Place your hand within the bounding box for calibration...")
    avg_x, avg_y, avg_z = calibrate_hand(cap)
    print("Calibration completed.")

    # Create the main application window
    root = tk.Tk()
    root.title("Hand Tracking")
    root.geometry("1280x600")  # Increase window height to accommodate the live XYZ counter

    # Create a canvas to display the black and white camera feed
    canvas_bw = tk.Canvas(root, width=640, height=480)
    canvas_bw.pack(side=tk.LEFT)

    # Create a canvas to display the hand tracking feed
    canvas_tracking = tk.Canvas(root, width=640, height=480)
    canvas_tracking.pack(side=tk.RIGHT)

    # Create a frame to hold the live coordinates display
    coord_frame = tk.Frame(root, bg="white", bd=5)
    coord_frame.pack(side=tk.TOP, pady=10)

    # Create a label to display live coordinates
    coord_label = tk.Label(coord_frame, text="Coordinates (X, Y, Z): ", font=("Helvetica", 16), bg="white")
    coord_label.pack()

    # Create a figure for the live graph
    fig = plt.figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    x_data, y_data, z_data = [], [], []
    line_x, = ax.plot(x_data, label='X')
    line_y, = ax.plot(y_data, label='Y')
    line_z, = ax.plot(z_data, label='Z')
    ax.set_xlim(0, 50)  # Adjust the x-axis range as needed
    ax.set_ylim(-2, 2)  # Adjust the y-axis range as needed
    ax.legend(loc='upper right')
    plt.xlabel('Time')
    plt.ylabel('Position')

    # Create a canvas to display the live graph
    canvas_graph = FigureCanvasTkAgg(fig, master=root)
    canvas_graph.draw()
    canvas_graph.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale for the black and white feed
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert the grayscale frame to RGB for hand tracking feed
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Process the frame to detect hands
        with mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands:
            results = hands.process(rgb_frame)

            # Draw landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Calculate the average position of the hand landmarks
                    x, y, z = calculate_average_hand_position(hand_landmarks)

                    # Adjust coordinates based on calibration
                    x_diff = (x - avg_x) * 2.5  # Adjust scale factor as needed
                    y_diff = (y - avg_y) * 2.5  # Adjust scale factor as needed
                    z_diff = (z - avg_z) * 2.5  # Adjust scale factor as needed

                    # Display live coordinates (X, Y, Z) with difference from calibrated position
                    coord_label.config(text=f"Coordinates (X, Y, Z): {x_diff:.2f}, {y_diff:.2f}, {z_diff:.2f}")

        # Convert the frames to PIL format
        pil_frame_bw = Image.fromarray(gray_frame)
        pil_frame_tracking = Image.fromarray(frame)

        # Convert PIL images to ImageTk format
        img_bw = ImageTk.PhotoImage(image=pil_frame_bw)
        img_tracking = ImageTk.PhotoImage(image=pil_frame_tracking)

        # Update the canvas images
        canvas_bw.create_image(0, 0, anchor=tk.NW, image=img_bw)
        canvas_tracking.create_image(0, 0, anchor=tk.NW, image=img_tracking)

        # Update the Tkinter event loop
        root.update()

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):

            # Process the frame to detect hands
            with mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands:
                results = hands.process(rgb_frame)

                # Draw landmarks on the frame
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Calculate the average position of the hand landmarks
                        x, y, z = calculate_average_hand_position(hand_landmarks)

                        # Adjust coordinates based on calibration
                        x_diff = (x - avg_x) * 2.5  # Adjust scale factor as needed
                        y_diff = (y - avg_y) * 2.5  # Adjust scale factor as needed
                        z_diff = (z - avg_z) * 2.5  # Adjust scale factor as needed

                        # Display live coordinates (X, Y, Z) with difference from calibrated position
                        coord_label.config(text=f"Coordinates (X, Y, Z): {x_diff:.2f}, {y_diff:.2f}, {z_diff:.2f}")

                        # Update the live graph
                        x_data.append(x_diff)
                        y_data.append(y_diff)
                        z_data.append(z_diff)

                        if len(x_data) > 50:  # Keep only the last 50 data points for visualization
                            x_data = x_data[-50:]
                            y_data = y_data[-50:]
                            z_data = z_data[-50:]

                        line_x.set_ydata(x_data)
                        line_y.set_ydata(y_data)
                        line_z.set_ydata(z_data)

                        # Redraw the canvas
                        canvas_graph.draw()
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track_hands()
