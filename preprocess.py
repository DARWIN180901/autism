import cv2
import os

# Configuration
INPUT_DIR = 'dataset'
OUTPUT_DIR = 'dataset_cropped'
# OpenCV's built-in pre-trained face detector
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_dataset():
    # Create output directories if they don't exist
    for category in ['Autism', 'Non_Autism']:
        os.makedirs(os.path.join(OUTPUT_DIR, category), exist_ok=True)

    total_processed = 0
    faces_found = 0

    for category in ['Autism', 'Non_Autism']:
        folder_path = os.path.join(INPUT_DIR, category)
        output_folder = os.path.join(OUTPUT_DIR, category)
        
        print(f"Processing {category} folder...")
        
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            total_processed += 1
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = FACE_CASCADE.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # If a face is found, crop and save it
            if len(faces) > 0:
                # Grab the first face found
                x, y, w, h = faces[0]
                
                # Add a small margin around the face
                margin = int(max(w, h) * 0.1)
                y1 = max(y - margin, 0)
                y2 = min(y + h + margin, img.shape[0])
                x1 = max(x - margin, 0)
                x2 = min(x + w + margin, img.shape[1])
                
                cropped_face = img[y1:y2, x1:x2]
                
                # Save to the new directory
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, cropped_face)
                faces_found += 1

    print("--- Preprocessing Complete ---")
    print(f"Total images scanned: {total_processed}")
    print(f"Total faces successfully cropped and saved: {faces_found}")

if __name__ == '__main__':
    process_dataset()