import cv2
import os

def collect_images(class_name, num_images=50):
    # Create folder for the class
    save_path = os.path.join('dataset', class_name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print(f"Collecting images for class: {class_name}")
    print("Press 'c' to capture an image, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(f'Collecting {class_name}', frame)

        key = cv2.waitKey(1) & 0xFF

        # Capture image when 'c' is pressed
        if key == ord('c'):
            img_name = os.path.join(save_path, f"{class_name}_{count}.jpg")
            cv2.imwrite(img_name, frame)
            count += 1
            print(f"Captured: {img_name}")
            if count >= num_images:
                print(f"Collected {num_images} images for {class_name}")
                break

        # Quit if 'q' is pressed
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
# collect_images('bottle', 50)
# collect_images('phone', 50)
# collect_images('cup', 50)
collect_images('background', 50)
