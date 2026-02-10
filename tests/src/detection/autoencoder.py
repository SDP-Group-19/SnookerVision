from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input  # type: ignore
from keras.models import Model, load_model  # type: ignore
import cv2
import numpy as np
import logging
import os

from src.core import config, state

logger = logging.getLogger(__name__)


class AutoEncoder:
    def __init__(self):
        self.autoencoder = self._build_autoencoder()
        self.detection_buffer = np.array([])
        self.obstruction_already_sent = False

    def _build_autoencoder(self):
        if os.path.exists(config.autoencoder_model_path):
            logger.info("Loading existing autoencoder model.")
            return load_model(config.autoencoder_model_path)

        input_img = Input(shape=(128, 128, 3))
        x = self._build_encoder(input_img)

        decoded = self._build_decoder(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.summary()

        clean_images = self.load_images()
        if len(clean_images) == 0:
            logger.error("No clean images found for training.")
            return None

        logger.info("Training autoencoder model.")
        autoencoder.fit(
            clean_images,
            clean_images,
            epochs=50,
            batch_size=32)

        autoencoder.save(config.autoencoder_model_path)
        logger.info("Autoencoder model saved.")

        return autoencoder

    def _build_encoder(self, input_image):
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_image)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        return x

    def _build_decoder(self, x):
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        return Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    def load_images(self):
        images = []
        logger.info(config.ae_data_path)
        for filename in os.listdir(config.ae_data_path):
            if filename.endswith(".jpg"):
                img = cv2.imread(os.path.join(config.ae_data_path, filename))
                img = cv2.resize(img, (128, 128))
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    images.append(img)
                else:
                    logger.warning(f"Failed to load image {filename}.")

        if not images:
            logger.error("No images found in the specified directory.")
            return None

        logger.info(f"Loaded {len(images)} images for training.")
        return np.array(images, dtype=np.float32) // 255.0

    def detect_obstruction(self, table_only):
        if table_only is None or self.autoencoder is None:
            logger.error(
                "Cannot detect obstruction: missing table data or model")
            return False

        # Preprocess image more efficiently
        obstruction = cv2.resize(table_only, (128, 128))
        obstruction = obstruction.astype(np.float32) / 255.0
        obstruction = obstruction[np.newaxis, ...]  # Faster than expand_dims

        # Get prediction with reduced verbosity
        reconstructed = self.autoencoder.predict(obstruction, verbose=0)

        # Vectorized MSE calculation
        mse = np.mean(np.square(obstruction - reconstructed))

        return self._update_and_check_buffer(mse) > config.obstruction_threshold

    def _update_and_check_buffer(self, mse):
        # Use a fixed-size buffer for better performance
        if not hasattr(self, 'buffer_index'):
            self.buffer_index = 0
            if len(self.detection_buffer) == 0:
                self.detection_buffer = np.zeros(
                    config.obstruction_buffer_size)

        # Update buffer using circular indexing
        self.detection_buffer[self.buffer_index] = mse
        self.buffer_index = (self.buffer_index +
                             1) % config.obstruction_buffer_size

        # Calculate mean only on filled values
        mean = np.mean(self.detection_buffer)

        # Check threshold only when needed
        threshold_diff = abs(mean - config.obstruction_threshold)
        warn_threshold = config.obstruction_threshold * config.obstruction_warn_if_within
        if threshold_diff <= warn_threshold:
            logger.warning(
                f"Mean is within {config.obstruction_warn_if_within * 100:.1f}% "
                f"of threshold: {mean:.6f}")

        return mean

    def handle_obstruction_detection(self, table_only):
        obstruction_detected = self.detect_obstruction(table_only)
        if obstruction_detected:
            if config.use_networking and state.network \
                    and not self.obstruction_already_sent:

                state.network.send_obstruction("true")
                self.obstruction_already_sent = True
            elif not config.use_networking:
                logger.warning("Obstruction detected!")
        elif not obstruction_detected and self.obstruction_already_sent:
            if config.use_networking and state.network:
                state.network.send_obstruction("false")
            self.obstruction_already_sent = False
