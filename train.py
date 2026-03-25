import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
import os

# Define dataset paths
train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'
os.makedirs('model', exist_ok=True)

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)

# Get the number of classes dynamically
num_classes = len(train_generator.class_indices)
print(f"Detected {num_classes} classes: {train_generator.class_indices}")

# Save class indices to a JSON file so app.py can use exactly the same mapping
class_indices_inv = {v: k for k, v in train_generator.class_indices.items()}
with open('model/class_indices.json', 'w') as f:
    json.dump(class_indices_inv, f)
print("Class indices saved to model/class_indices.json")

# Load MobileNetV2 
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# FINE-TUNING: Unfreeze the top layers of the base model
base_model.trainable = True
# Freeze the first 100 layers, let the rest train
for layer in base_model.layers[:100]:
    layer.trainable = False

# Build custom classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x) # Slightly increased dropout to prevent overfitting
output = Dense(num_classes, activation='softmax')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=output)

# Compile model (Using a slightly lower learning rate since we are fine-tuning)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Define callbacks
callbacks = [
    # Increased patience so it doesn't stop too early
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

# Train the model (Increased epochs to 30)
history = model.fit(
    train_generator, 
    epochs=30, 
    validation_data=validation_generator, 
    callbacks=callbacks
)

# Save the model
model.save('model/optimized_model.h5')

# ... (Keep your existing plotting and accuracy print code here) ...