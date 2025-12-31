# graph.py
import networkx as nx

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class KnowledgeGraphBuilder:
    def __init__(self):
        self.enabled = False
        self.nlp = None

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.blank("en")  # ✅ safe for Streamlit Cloud
                self.enabled = True
            except Exception:
                self.enabled = False

    def build_graph(self, text):
        G = nx.DiGraph()

        if not self.enabled or not text.strip():
            return G

        words = text.split()[:6]  # lightweight entity simulation
        for w in words:
            G.add_node(w, label="Entity")

        return G

    def extract_features(self, text):
        G = self.build_graph(text)
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()
        density = nx.density(G) if nodes > 1 else 0.0
        return [nodes, edges, density]
