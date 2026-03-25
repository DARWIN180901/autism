import os
import cv2
import shutil
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# --- 1. DIRECTORY CONFIGURATION ---
RAW_VIDEO_DIR = 'video_dataset'
EXTRACTED_FRAMES_DIR = 'video_frames_dataset' # Dedicated folder for video frames

train_video_dir = os.path.join(RAW_VIDEO_DIR, 'train')
validation_video_dir = os.path.join(RAW_VIDEO_DIR, 'validation')

train_frames_dir = os.path.join(EXTRACTED_FRAMES_DIR, 'train')
validation_frames_dir = os.path.join(EXTRACTED_FRAMES_DIR, 'validation')

# --- 2. VIDEO PREPROCESSING LOGIC ---
def process_videos_to_frames(source_dir, dest_dir, frame_skip=30):
    if not os.path.exists(source_dir):
        print(f"Warning: Directory {source_dir} not found. Skipping extraction.")
        return

    print(f"Extracting frames from videos in {source_dir}...")
    for class_name in os.listdir(source_dir):
        class_video_path = os.path.join(source_dir, class_name)
        class_image_path = os.path.join(dest_dir, class_name)
        
        if not os.path.isdir(class_video_path):
            continue
            
        # Create destination directory for this class
        os.makedirs(class_image_path, exist_ok=True)

        for video_name in os.listdir(class_video_path):
            if not video_name.lower().endswith(('.mp4', '.avi', '.mov')):
                continue
                
            video_path = os.path.join(class_video_path, video_name)
            cap = cv2.VideoCapture(video_path)
            
            frame_count = 0
            saved_count = 0
            base_filename = video_name.rsplit('.', 1)[0]
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save 1 frame every 'frame_skip' frames
                if frame_count % frame_skip == 0:
                    output_filename = f"{base_filename}_f{frame_count}.jpg"
                    output_path = os.path.join(class_image_path, output_filename)
                    if not os.path.exists(output_path):
                        cv2.imwrite(output_path, frame)
                        saved_count += 1
                        
                frame_count += 1
                
            cap.release()
            if saved_count > 0:
                print(f"  -> Saved {saved_count} frames from {video_name}")

print("--- Step 1: Extracting Frames from Training Videos ---")
process_videos_to_frames(train_video_dir, train_frames_dir, frame_skip=30)

print("\n--- Step 2: Extracting Frames from Validation Videos ---")
process_videos_to_frames(validation_video_dir, validation_frames_dir, frame_skip=30)
print("--- Video Extraction Complete ---\n")

# --- 3. MODEL TRAINING ON VIDEO FRAMES ---
# Data augmentation (helps the model generalize video frames better)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load images from the newly created video frames directories
train_generator = train_datagen.flow_from_directory(
    train_frames_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_frames_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

num_classes = len(train_generator.class_indices)
print(f"Detected {num_classes} classes: {train_generator.class_indices}")

# Build the MobileNetV2 Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

print("\n--- Step 3: Training the Video-Specific Model ---")
history = model.fit(train_generator, epochs=20, validation_data=validation_generator, callbacks=callbacks)

# Save as a distinct model file
os.makedirs('model', exist_ok=True)
model.save('model/video_optimized_model.h5')
print("\nModel saved successfully as 'model/video_optimized_model.h5'")

# --- 4. PLOT RESULTS ---
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(train_acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Video Model: Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Video Model: Loss")

plt.show()