import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from orgmim.trainers import pretrain_provider


class PretrainProviderTest(unittest.TestCase):
    def test_normalizes_images_before_intensity_augmentation(self):
        raw = np.arange(32, 96, dtype=np.uint8).reshape(4, 4, 4)
        mam = np.full_like(raw, 128)
        with tempfile.TemporaryDirectory() as directory:
            groups = Path(directory) / "groups.json"
            groups.write_text(json.dumps({"0": ["sample.tif"]}), encoding="utf-8")
            dataset = pretrain_provider.Train(
                SimpleNamespace(
                    MODEL=SimpleNamespace(img_size=4),
                    DATA=SimpleNamespace(data_folder="raw", mam_folder="mam"),
                    TRAIN=SimpleNamespace(group_path=groups),
                )
            )
            dataset.simple_aug.skip_ratio = 0.0
            for skip_ratio in (0.0, 1.0):
                with self.subTest(augmentation_enabled=skip_ratio == 1.0):
                    dataset.intensity_aug.skip_ratio = skip_ratio
                    # Enable the clipping path with neutral contrast, brightness, and gamma.
                    with patch.object(np.random, "rand", return_value=0.5):
                        with patch.object(pretrain_provider.imageio, "volread", side_effect=[raw, mam]):
                            images, attention = dataset[0]

                    self.assertEqual(images.dtype, np.float32)
                    np.testing.assert_array_equal(images[0], raw.astype(np.float32) / 255.0)
                    np.testing.assert_array_equal(attention[0], mam.astype(np.float32) / 255.0)


if __name__ == "__main__":
    unittest.main()
