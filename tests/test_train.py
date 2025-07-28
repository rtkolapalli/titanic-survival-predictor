import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import train

def test_model_file_created():
    train.train_and_save()
    assert os.path.exists("models/titanic_model.pkl")
