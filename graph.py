# graph.py
import networkx as nx

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class KnowledgeGraphBuilder:
    def __init__(self, model="en_core_web_sm"):
        self.enabled = False
        self.nlp = None

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model)
                self.enabled = True
            except Exception:
                # spaCy model not available on Streamlit Cloud
                self.enabled = False

    def build_graph(self, text):
        G = nx.DiGraph()

        if not self.enabled or not text.strip():
            return G

        doc = self.nlp(text)
        for ent in doc.ents:
            G.add_node(ent.text, label=ent.label_)

        return G

    def extract_features(self, text):
        if not self.enabled:
            return [0, 0, 0.0]

        G = self.build_graph(text)
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()
        density = nx.density(G) if nodes > 1 else 0.0
        return [nodes, edges, density]
