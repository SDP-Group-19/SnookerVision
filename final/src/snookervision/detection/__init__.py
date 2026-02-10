from .detection import DetectionModel

try:
    from .autoencoder import AutoEncoder
except Exception:  # Optional dependency (tensorflow/keras)
    AutoEncoder = None
