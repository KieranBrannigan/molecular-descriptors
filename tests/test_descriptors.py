import unittest

class TestDescriptors(unittest.TestCase):

    def test_num_of_phosphates(self):
        from y4_python.python_modules.descriptors import num_of_phosphate_bonds
        testSmiles = "CS(=O)C(C)(C)C1CC1C(c2ccccc2)P(C(C=O)C)(O)=O"

        expected = 1 # we expect to get 1 phosphate
        result = num_of_phosphate_bonds(testSmiles)
        self.assertEqual(expected, result)

        "O=P(Oc1ccc(OP(=O)(N2CC2)N2CC2)cc1)(N1CC1)N1CC1"
        testSmiles = "O=P(Oc1ccc(OP(=O)(N2CC2)N2CC2)cc1)(N1CC1)N1CC1"

        expected = 2 # we expect to get 1 phosphate
        result = num_of_phosphate_bonds(testSmiles)
        self.assertEqual(expected, result)

    def test_num_of_sulfates(self):
        from y4_python.python_modules.descriptors import num_of_sulfate_bonds
        testSmiles = "CS(=O)C(C)(C)C1CC1C(c2ccccc2)P(C(C=O)C)(O)=O"

        expected = 1 # we expect to get 1 sulfate
        result = num_of_sulfate_bonds(testSmiles)
        self.assertEqual(expected, result)

        testSmiles = "C=CCSC/C(O)=N/S(=O)(=O)c1cnn(C)c1"
        expected = 2 # we expect to get 2 sulfates
        result = num_of_sulfate_bonds(testSmiles)
        self.assertEqual(expected, result)

    def test_get_end_of_branch_idx(self):
        from y4_python.python_modules.descriptors import get_end_of_branch_idx

        test_string_with_branch = "CS(=O)C(C)(C)C1CC1C(c2ccccc2)P(C(C=O)C)(O)=O"
        start_branch_idx = 2
        expected = 5
        result = get_end_of_branch_idx(test_string_with_branch, start_branch_idx)
        self.assertEqual(expected, result)

        test_string_with_branch = "P(C(C=O)C)(O)=O"
        start_branch_idx = 1
        expected = 9
        result = get_end_of_branch_idx(test_string_with_branch, start_branch_idx)
        self.assertEqual(expected, result)
        
    def test_num_of_atoms(self):
        from y4_python.python_modules.descriptors import num_of_atoms
        testSmiles = "C=CCSC/C(O)=N/S(=O)(=O)c1cnn(C)c1"
        result = num_of_atoms(testSmiles, ["S", "N"])
        expected = 2+3 # we expect 2 S and 3 N
        self.assertEqual(expected, result)

if __name__ == "__main__":
    unittest.main()