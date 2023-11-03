from ultralytics import YOLO
import cv2

# Webcam Video Capture
#cap = cv2.VideoCapture(0)
#cap.set(3, 1280)
#cap.set(4, 720)

# Video File Capture
cap = cv2.VideoCapture("../Videos/bikes.mp4")

model = YOLO("../Yolo-Weights/yolov8n.pt")

ClassNames = [
    "Person", "Bicycle", "Car", "Motorcycle", "Airplane",
    "Bus", "Train", "Truck", "Boat", "Traffic Light",
    "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench", "Bird",
    "Cat", "Dog", "Horse", "Sheep", "Cow",
    "Elephant", "Bear", "Zebra", "Giraffe", "Backpack",
    "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee",
    "Skis", "Snowboard", "Sports Ball", "Kite", "Baseball Bat",
    "Baseball Glove", "Skateboard", "Surfboard", "Tennis Racket", "Bottle",
    "Wine Glass", "Cup", "Fork", "Knife", "Spoon",
    "Bowl", "Banana", "Apple", "Sandwich", "Orange",
    "Broccoli", "Carrot", "Hot Dog", "Pizza", "Donut",
    "Cake", "Chair", "Couch", "Potted Plant", "Bed",
    "Dining Table", "Toilet", "Tv", "Laptop", "Mouse",
    "Remote", "Keyboard", "Cell Phone", "Microwave", "Oven",
    "Toaster", "Sink", "Refrigerator", "Book", "Clock",
    "Vase", "Scissors", "Teddy Bear", "Hair Drier", "Toothbrush"
]


# Function to draw labeled rectangle
def draw_label(img, text, position, bg_color=(0, 255, 0), text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1):
    x, y = position
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    if y - text_h - 10 < 0:  # If the text is at the top border
        cv2.rectangle(img, (x, y + 10), (x + text_w, y + text_h + 10), bg_color, -1)
        cv2.putText(img, text, (x, y + text_h + 10), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y), bg_color, -1)
        cv2.putText(img, text, (x, y - 10), font, font_scale, text_color, font_thickness, cv2.LINE_AA)


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for result in results:
        bounding_boxes = result.boxes
        for bounding_box in bounding_boxes:
            tensor_x1, tensor_y1, tensor_x2, tensor_y2 = bounding_box.xyxy[0]
            x1, y1, x2, y2 = int(tensor_x1), int(tensor_y1), int(tensor_x2), int(tensor_y2)

            # Draw the bounding box
            cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=3)

            # Get confidence and class name
            confidence = round(float(bounding_box.conf[0]), 2)
            class_id = int(bounding_box.cls[0])
            class_name = ClassNames[class_id] if class_id < len(ClassNames) else 'Unknown'

            # Label format
            label = f'{class_name}: {confidence:.2f}'
            # Draw the label
            draw_label(img, label, position=(x1, y1), text_color=(0, 0, 0), font_thickness=1, font_scale=0.7)

    cv2.imshow("Yolo Webcam", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()