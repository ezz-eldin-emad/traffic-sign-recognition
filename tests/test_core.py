import unittest

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import MODEL_REGISTRY
from src.evaluation.robustness import transform_images
from src.inference.model_loader import (
    load_custom_cnn,
    load_mobilenet,
    load_tflite_model,
)
from src.models.classical_ml import (
    ImagePreprocessor,
    build_random_forest_pipeline,
    build_svm_pipeline,
)
from src.inference.preprocessing import preprocess_image
from src.xai.gradcam import make_gradcam_heatmap
from src.xai.targets import get_target_layer


class CoreSmokeTests(unittest.TestCase):
    def test_registry_contains_six_models(self):
        self.assertEqual(
            set(MODEL_REGISTRY),
            {
                "custom_cnn",
                "mobilenet_v3_small_gtsrb",
                "custom_cnn_float16_tflite",
                "custom_cnn_int8_tflite",
                "svm",
                "random_forest",
            },
        )

    def test_preprocessing_returns_batched_tensor(self):
        image = Image.new("RGB", (24, 32), color=(10, 20, 30))
        tensor = preprocess_image(image, (128, 128))
        self.assertEqual(tuple(tensor.shape), (1, 128, 128, 3))
        self.assertEqual(tensor.dtype, tf.float32)

    def test_robustness_transforms_keep_batch_shape(self):
        images = tf.zeros((2, 32, 32, 3), dtype=tf.float32)
        for transform_name in ("blur", "illumination", "perspective"):
            transformed = transform_images(images, transform_name, 2)
            self.assertEqual(tuple(transformed.shape), (2, 32, 32, 3))

    def test_gradcam_works_for_saved_models(self):
        for model, shape in (
            (load_custom_cnn(), (128, 128, 3)),
            (load_mobilenet(), (224, 224, 3)),
        ):
            image = tf.zeros((1,) + shape, dtype=tf.float32)
            heatmap = make_gradcam_heatmap(image, model, get_target_layer(model))
            self.assertEqual(heatmap.ndim, 2)
            self.assertTrue(np.isfinite(heatmap).all())
            self.assertGreaterEqual(float(heatmap.min()), 0.0)
            self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_tflite_models_return_probabilities(self):
        for model_name in ("custom_cnn_float16_tflite", "custom_cnn_int8_tflite"):
            model = load_tflite_model(model_name)
            probabilities = model.predict(np.zeros((1, 128, 128, 3), dtype=np.float32))
            self.assertEqual(tuple(probabilities.shape), (1, 43))
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=3)

    def test_classical_pipelines_match_notebook_preprocessing(self):
        images = np.zeros((2, 24, 32, 3), dtype=np.uint8)
        processed = ImagePreprocessor().fit_transform(images)

        self.assertEqual(tuple(processed.shape), (2, 100, 100))
        self.assertEqual(
            list(build_svm_pipeline().named_steps),
            ["preprocess", "hog", "scaler", "clf"],
        )
        self.assertEqual(
            list(build_random_forest_pipeline().named_steps),
            ["preprocess", "flatten", "clf"],
        )


if __name__ == "__main__":
    unittest.main()
