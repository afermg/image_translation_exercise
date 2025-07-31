"""
Download the datasets automatically, before any analysis is performed.
"""

from image_translation.data import get_data

for i in ("movie", "train", "test", "training_logs"):
    print(get_data(i))
