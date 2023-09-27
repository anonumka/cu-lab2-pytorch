import unittest
import torch
import numpy as np
from torch.utils.cpp_extension import load


class LabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ext = load(
            name='my_extension',
            sources=['relu.cu'],
            extra_cuda_cflags=['-O2'],
            extra_cflags=['-O2'],
        )

    def test_relu(self):
        n = torch.randint(size=(1,), low=1, high=2048)

        x = torch.rand((n,), device='cuda')
        z = LabTest.ext.my_relu(x)

        # z_ = x * (x > 0).float()
        z_ = torch.nn.functional.relu(x)

        self.assertTrue(torch.allclose(z, z_, atol=1e-7, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
