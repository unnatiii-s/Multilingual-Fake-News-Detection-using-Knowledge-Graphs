<<<<<<< HEAD

from transformers import pipeline
import sys
import traceback

print("Attempting to load model...")
try:
    classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
    print("Model loaded successfully!")
except Exception:
    print("Failed to load model.")
    traceback.print_exc()
=======

from transformers import pipeline
import sys
import traceback

print("Attempting to load model...")
try:
    classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
    print("Model loaded successfully!")
except Exception:
    print("Failed to load model.")
    traceback.print_exc()
>>>>>>> b916bcb10db7c6ea06e249b9569f284f9348ebd0
