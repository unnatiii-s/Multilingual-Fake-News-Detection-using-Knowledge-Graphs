
import networkx as nx
import spacy

class KnowledgeGraphBuilder:
    def __init__(self, model="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            # Fallback or download if missing (though strictly we should check before)
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            self.nlp = spacy.load(model)

    def build_graph(self, text):
        doc = self.nlp(text)
        G = nx.DiGraph()
        
        # Add entities (Nodes)
        for ent in doc.ents:
            G.add_node(ent.text, label=ent.label_)
            
        # Add simple relations based on dependency tree (Edges)
        # This is a simplified heuristic: linking Subject -> Verb -> Object
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subj = token
                verb = token.head
                # Find object
                objs = [child for child in verb.children if child.dep_ in ("dobj", "pobj", "attr")]
                for obj in objs:
                     G.add_edge(subj.text, obj.text,  relation=verb.text)
                     
                     # Also ensure nodes exist even if they are not NER
                     if not G.has_node(subj.text):
                         G.add_node(subj.text, label="TOKEN")
                     if not G.has_node(obj.text):
                         G.add_node(obj.text, label="TOKEN")
                         
        return G

    def extract_features(self, text):
        """
        Extracts graph statistics as simple numerical features.
        Returns: [num_nodes, num_edges, density]
        """
        G = self.build_graph(text)
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G) if num_nodes > 1 else 0.0
        return [num_nodes, num_edges, density]
