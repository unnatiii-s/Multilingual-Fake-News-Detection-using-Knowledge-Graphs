# src/model.py
# Deployment-safe lightweight model (NO torch, NO transformers)

class FakeNewsModel:
    def __init__(self, use_kg=True):
        self.use_kg = use_kg

        # Simple heuristic keywords (demo inference layer)
        self.fake_keywords = [
            "secret", "miracle", "cure", "shocking",
            "hoax", "conspiracy", "instantly", "confirmed",
            "scientists claim", "breaking", "guaranteed"
        ]

    def predict_proba(self, text):
        """
        Returns probability of fake news (0–1)
        """
        if not text or not text.strip():
            return 0.5

        text = text.lower()

        score = sum(word in text for word in self.fake_keywords)
        probability = score / len(self.fake_keywords)

        # clamp value for stability
        return max(0.05, min(probability, 0.95))
