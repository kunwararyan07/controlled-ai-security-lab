import unittest

from core.models import MockModel, ModelAdapter


class TestMockModel(unittest.TestCase):
    def test_mock_model_instantiation(self):
        model = MockModel(response="test response")
        self.assertIsInstance(model, MockModel)
        self.assertIsInstance(model, ModelAdapter)

    def test_mock_model_generate_returns_expected_response(self):
        expected = "deterministic output"
        model = MockModel(response=expected)
        output = model.generate("arbitrary prompt")
        self.assertEqual(output, expected)

    def test_mock_model_generate_returns_string(self):
        model = MockModel(response="hello world")
        output = model.generate("test prompt")
        self.assertIsInstance(output, str)


if __name__ == "__main__":
    unittest.main()
