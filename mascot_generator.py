import cv2
import numpy as np

def overlay_graduation_cap_blended(mascot_path, cap_path, cascade_path, output_path):
    # Load main image and convert to OpenCV format
    mascot = cv2.imread(mascot_path)
    if mascot is None:
        print("Could not load mascot image.")
        return

    # Load cap image with alpha channel
    cap = cv2.imread(cap_path, cv2.IMREAD_UNCHANGED)
    if cap is None:
        print("Could not load graduation cap image.")
        return

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(mascot, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No face detected.")
        return

    for (x, y, w, h) in faces:
        # Resize the cap image to match the width of the face
        cap_width = w
        cap_height = int(h * 0.6)
        cap_resized = cv2.resize(cap, (cap_width, cap_height), interpolation=cv2.INTER_AREA)

        if cap_resized.shape[2] == 4:
            cap_rgb = cap_resized[:, :, :3]
            cap_alpha = cap_resized[:, :, 3]
            mask = cv2.merge([cap_alpha, cap_alpha, cap_alpha])
        else:
            print("⚠️ Cap image has no alpha channel — creating solid mask.")
            cap_rgb = cap_resized
            mask = 255 * np.ones_like(cap_rgb)


        # Define position (center point) above the head
        center_x = x + cap_width // 2
        center_y = y - cap_height // 2 + 10
        center = (center_x, center_y)

        # Use seamlessClone for realistic blending
        mascot = cv2.seamlessClone(cap_rgb, mascot, mask, center, cv2.MIXED_CLONE)

    # Save final result
    cv2.imwrite(output_path, mascot)
    print(f"✅ Saved realistic image as {output_path}")



overlay_graduation_cap_blended(
    mascot_path="mascot.jpg",
    cap_path="grad_cap.jpg",
    cascade_path="haarcascade_frontalface_default.xml",
    output_path="mascot_graduated_blended.jpg"
)
