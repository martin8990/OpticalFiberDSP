import unittest
from mimo.mimo import CMAErrorCalculator, FrequencyDomainBlockwizeMimo 
import numpy as np

class TestStringMethods(unittest.TestCase):
    def test_mimo_only_accepts_complex(self):
        mimo = FrequencyDomainBlockwizeMimo(2)
        sig = np.zeros(10000).reshape(2,5000)
        self.assertRaises(ValueError,mimo.equalize_signal,sig)
        sig.dtype = np.complex128
        mimo.equalize_signal(sig)
        
if __name__ == '__main__':
    unittest.main()