# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Created on Mon Nov 11 16:05:34 2024

@author: acer
"""
import unittest
from parkinsons_model import train_model, evaluate_model
import numpy as np

class TestParkinsonsModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This method runs once before all tests
        # Set up dummy data for testing
        cls.dummy_data = {
            'features': np.array([[0.2, 0.4, 0.6], [0.5, 0.8, 0.3]]),
            'labels': np.array([1, 0])
        }

    def test_train_model(self):
        """Test if train_model returns a model instance."""
        model = train_model(self.dummy_data['features'], self.dummy_data['labels'])
        
        # Check if model is not None
        self.assertIsNotNone(model, "Model should not be None after training")
        
        # Optionally, check if model has the expected attributes/methods
        self.assertTrue(hasattr(model, 'predict'), "Model should have a 'predict' method")

    def test_evaluate_model(self):
        """Test if evaluate_model returns an accuracy value."""
        # Train a model to evaluate
        model = train_model(self.dummy_data['features'], self.dummy_data['labels'])
        
        # Perform evaluation
        accuracy = evaluate_model(model, self.dummy_data['features'], self.dummy_data['labels'])
        
        # Check if accuracy is a float between 0 and 1
        self.assertIsInstance(accuracy, float, "Accuracy should be a float")
        self.assertGreaterEqual(accuracy, 0.0, "Accuracy should be >= 0")
        self.assertLessEqual(accuracy, 100.0, "Accuracy should be <= 100")

if __name__ == '__main__':
    unittest.main()
<<<<<<< HEAD
=======




    
>>>>>>> ad00a94 (Add and remove files as needed for the project)
