import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def build_model(num_classes=7):
    """
    Builds a CNN model for facial emotion recognition.
    Expected input shape is 48x48 grayscale images (FER2013 standard).
    """
    model = Sequential([
        # 1st Convolutional Layer
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # 2nd Convolutional Layer
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # 3rd Convolutional Layer
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Flattening
        Flatten(),

        # Fully Connected Layer
        Dense(256, activation='relu'),
        Dropout(0.5),

        # Output Layer
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(train_dir, val_dir, epochs=50, batch_size=64):
    """
    Trains the CNN model using images from directories.
    Requires FER2013 dataset downloaded and extracted into train/val folders.
    """
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: Datasets not found at {train_dir} or {val_dir}.")
        print("Please download FER2013 dataset and place it in the datasets folder.")
        return

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical'
    )

    model = build_model(num_classes=train_generator.num_classes)
    model.summary()

    # Train
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size
    )

    # Save the model
    os.makedirs('../models', exist_ok=True)
    model.save('../models/face_model.h5')
    print("Model saved to ../models/face_model.h5")

if __name__ == "__main__":
    print("--- Emotion Detection CNN Training ---")
    # Uncomment and set correct paths when dataset is downloaded
    # train_dir = '../datasets/fer2013/train'
    # val_dir = '../datasets/fer2013/test'
    # train_model(train_dir, val_dir, epochs=30)
