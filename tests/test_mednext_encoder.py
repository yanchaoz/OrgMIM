import unittest
from unittest.mock import patch

import torch

from orgmim.mim_cnn import encoder3D
from orgmim.mim_cnn.mednext_encoder import MedNeXtEncoder


class MedNeXtTest(unittest.TestCase):
    def test_feature_hierarchy(self):
        expected = [
            (1, 32, 32, 32, 32), (1, 64, 16, 16, 16), (1, 128, 8, 8, 8),
            (1, 256, 4, 4, 4), (1, 512, 2, 2, 2),
        ]
        for scale in ("small", "base", "medium", "large"):
            with self.subTest(scale=scale), torch.no_grad():
                features = MedNeXtEncoder(scale)(torch.zeros(1, 1, 32, 32, 32), hierarchical=True)
                self.assertEqual([tuple(feature.shape) for feature in features], expected)

    def test_sparse_group_norm_uses_active_voxels_per_sample(self):
        mask = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1], dtype=torch.bool)
        mask = torch.stack((mask, ~mask)).reshape(2, 1, 2, 2, 2)
        x = torch.arange(32, dtype=torch.float32).reshape(2, 2, 2, 2, 2)
        with patch.object(encoder3D, "_cur_active", mask):
            out = encoder3D.SparseGroupNorm(2, 2)(x)
        active = mask.expand_as(x)
        visible = out[active].reshape(2, 2, -1)
        torch.testing.assert_close(visible.mean(dim=-1), torch.zeros(2, 2))
        torch.testing.assert_close(visible.var(dim=-1, unbiased=False), torch.ones(2, 2))
        self.assertTrue((out[~active] == 0).all())


if __name__ == "__main__":
    unittest.main()
