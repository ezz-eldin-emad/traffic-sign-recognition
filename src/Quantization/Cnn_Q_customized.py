"""Post-training int8 quantization for the custom CNN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import tensorflow as tf
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DIR, TRAIN_DIR
model = tf.keras.models.load_model("/var/home/alipc/Documents/GitHub/traffic-sign-recognition/models/cnn_model_customized.keras")





IMAGE_SIZE = (128, 128)
DEFAULT_MODEL_PATH = MODEL_DIR / "cnn_model_customized.keras"

'''
define a representative dataset generator for the quantization process. 
    This function yields a specified number of images from the provided data directory,
    which are used to calibrate the activation ranges during the quantization of the model.
    The images are resized to a defined IMAGE_SIZE and converted to float32 tensors.
        The function ensures that the data directory exists and raises an error if it does not.'''
def representative_dataset(
    data_dir: Path, sample_count: int
) -> Iterator[list[tf.Tensor]]:
    """Yield float32 images used to calibrate int8 activation ranges."""
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Representative data directory not found: {data_dir}. "
            "Prepare the GTSRB training data or pass --data-dir."
        )

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=1,
        shuffle=False,
    )
    for index, images in enumerate(dataset):
        if index >= sample_count:
            break
        yield [tf.cast(images, tf.float32)]

'''
    convert a saved Keras model to a float16 or full-int8 or full-float16 TFLite model.
'''
def quantize_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    output_path: Path = None,
    data_dir: Path = TRAIN_DIR,
    sample_count: int = 100,
    mode: str = "int8",
) -> Path:
    """Convert a saved Keras model to a float16 or full-int8 TFLite model."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Keras model not found: {model_path}")
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")

    output_path = output_path or model_path.with_name(
        f"{model_path.stem}_{mode}.tflite"
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if mode == "float16":
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "int8":
        converter.representative_dataset = lambda: representative_dataset(
            data_dir, sample_count
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        raise ValueError("mode must be 'int8' or 'float16'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(converter.convert())
    return output_path

''' Model quantization can be run from the command line or as a Python function.
 The following command will quantize the model to int8 or float16 and save it to the default location to model folders

'''
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--mode", choices=("int8", "float16"), default="int8")
    args = parser.parse_args()
    output = quantize_model(
        model_path=args.model,
        output_path=args.output,
        data_dir=args.data_dir,
        sample_count=args.samples,
        mode=args.mode,
    )
    print(f"Saved {args.mode} quantized model to {output}")


if __name__ == "__main__":
    main()
