import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

# ===============================
# TPU Initialization
# ===============================
try:
    # Detect TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # automatically detects TPU in Cloud TPU VM
    print("TPU detected:", tpu.cluster_spec().as_dict())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU with", strategy.num_replicas_in_sync, "accelerators.")
except Exception as e:
    print("TPU not detected or initialization failed; falling back to default strategy. Error:", e)
    strategy = tf.distribute.get_strategy()
    print("Running on", strategy.num_replicas_in_sync, "accelerators.")

# ===============================
# Data Loader and Generator
# ===============================

def load_dataset():
    """
    Loads video file paths and corresponding labels.
    Folder "./dataset/sludge" videos are labeled 1.
    Folder "./dataset/non_sludge" videos are labeled 0.
    """
    video_paths = []
    labels = []

    sludge_dir = "./dataset/sludge"
    non_sludge_dir = "./dataset/non_sludge"

    # Load sludge videos (label 1)
    for fname in os.listdir(sludge_dir):
        if fname.lower().endswith('.mp4'):
            video_paths.append(os.path.join(sludge_dir, fname))
            labels.append(1)

    # Load non-sludge videos (label 0)
    for fname in os.listdir(non_sludge_dir):
        if fname.lower().endswith('.mp4'):
            video_paths.append(os.path.join(non_sludge_dir, fname))
            labels.append(0)

    return video_paths, labels

class VideoDataGenerator(Sequence):
    """
    Keras Sequence to load videos on the fly.
    For each video, it uniformly samples a fixed number of frames,
    resizes them, converts BGR to RGB, and normalizes pixel values.
    """
    def __init__(self, video_paths, labels, batch_size=4, num_frames=16, target_size=(224, 224), shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_video_paths = self.video_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_video_paths, batch_labels)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.video_paths, self.labels))
            np.random.shuffle(combined)
            self.video_paths, self.labels = zip(*combined)
            self.video_paths = list(self.video_paths)
            self.labels = list(self.labels)

    def __data_generation(self, batch_video_paths, batch_labels):
        batch_data = []
        for video_path in batch_video_paths:
            frames = self.load_video(video_path)
            batch_data.append(frames)
        X = np.array(batch_data)
        y = np.array(batch_labels, dtype=np.float32)
        return X, y

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # If video has fewer frames than desired, repeat the last frame.
        if total_frames < self.num_frames:
            frame_indices = np.concatenate([np.arange(total_frames),
                                            np.full((self.num_frames - total_frames,), total_frames - 1)])
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=np.int32)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                # If frame reading fails, use a black frame.
                frame = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            frame = cv2.resize(frame, self.target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        cap.release()
        return np.array(frames)

# ===============================
# Model Definition
# ===============================
def build_model(num_frames=16, target_size=(224, 224)):
    """
    Builds a video classification model using a TimeDistributed EfficientNetB0 backbone
    and an LSTM for temporal aggregation. Dropout is added for regularization.
    """
    input_shape = (num_frames, target_size[1], target_size[0], 3)  # (frames, height, width, channels)
    inputs = layers.Input(shape=input_shape)

    # Pretrained CNN applied to each frame
    cnn_base = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
    cnn_base.trainable = False  # Freeze the backbone for transfer learning
    x = layers.TimeDistributed(cnn_base)(inputs)

    # Optional: Insert spatial attention modules here if needed

    # Temporal aggregation using LSTM
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)

    # Final classification layer
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# ===============================
# Main Training Pipeline
# ===============================
if __name__ == '__main__':
    # Load dataset
    video_paths, labels = load_dataset()
    print(f"Total videos found: {len(video_paths)}")

    # Split dataset: 70% train, 15% validation, 15% test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        video_paths, labels, test_size=0.3, random_state=42, stratify=labels)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

    print(f"Training videos: {len(train_paths)}")
    print(f"Validation videos: {len(val_paths)}")
    print(f"Testing videos: {len(test_paths)}")

    # Set parameters
    batch_size = 4
    num_frames = 16
    target_size = (224, 224)
    epochs = 50

    # Create generators
    train_generator = VideoDataGenerator(train_paths, train_labels, batch_size=batch_size, num_frames=num_frames, target_size=target_size, shuffle=True)
    val_generator = VideoDataGenerator(val_paths, val_labels, batch_size=batch_size, num_frames=num_frames, target_size=target_size, shuffle=False)
    test_generator = VideoDataGenerator(test_paths, test_labels, batch_size=batch_size, num_frames=num_frames, target_size=target_size, shuffle=False)

    # Build and compile the model within the TPU strategy scope
    with strategy.scope():
        model = build_model(num_frames=num_frames, target_size=target_size)

    model.summary()

    # Set up callbacks: EarlyStopping and ModelCheckpoint
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model_tpu.h5', monitor='val_loss', save_best_only=True)
    ]

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save the final model
    model.save('final_video_classifier_tpu.h5')
