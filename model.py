# model.py (DEPLOYMENT SAFE)

class FakeNewsModel:
    def __init__(self, use_kg=True):
        self.use_kg = use_kg

        # simple keyword-based heuristic (demo purpose)
        self.fake_keywords = [
            "secret", "miracle", "cure", "shocking",
            "hoax", "conspiracy", "instantly", "confirmed"
        ]

    def predict_proba(self, text):
        if not text:
            return 0.5

        text = text.lower()
        score = sum(word in text for word in self.fake_keywords) / len(self.fake_keywords)
        return min(max(score, 0.05), 0.95)
