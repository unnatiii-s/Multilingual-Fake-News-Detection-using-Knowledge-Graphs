import os

# Force transformers to use tf-keras if available / legacy keras behaviour
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
from graph import KnowledgeGraphBuilder
from model import FakeNewsModel
from transformers import AutoTokenizer
import torch

# --- Page Config ---
st.set_page_config(
    layout="wide", 
    page_title="Fake News Detector",
    page_icon="🕵️‍♂️"
)

# --- Custom CSS for Polish ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    h2, h3 {
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Models ---
@st.cache_resource
def load_models():
    # Text + KG Model
    try:
        model = FakeNewsModel("distilbert-base-multilingual-cased", use_kg=True)
        model.load_state_dict(torch.load("models/fake_news_model.pt", map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        model = None

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    kg_builder = KnowledgeGraphBuilder()
    
    return model, tokenizer, kg_builder

model, tokenizer, kg_builder = load_models()

st.set_page_config(layout="wide", page_title="Multilingual Fake News Detection")

st.title("Multilingual Fake News Detection with Knowledge Graphs")
st.markdown("""
This application uses **Supervised Multilingual DistilBERT** (Fine-tuned on FakeNewsNet) to detect fake news in multiple languages 
and visualizes the content as a **Knowledge Graph** to help identify key entities and relations.
""")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("📝 Input News")
    input_text = st.text_area("Enter news text here:", height=300, 
                              placeholder="e.g., The earth is flat and the moon is made of cheese...",
                              help="Supports 100+ languages.")
    
    if st.button("🔍 Analyze"):
        if model is None:
            st.error("⚠️ Model not loaded! Please run `python train.py` first.")
        elif not input_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("🧠 Processing..."):
                # 1. Tokenize
                encoding = tokenizer.encode_plus(
                    input_text,
                    max_length=128,
                    add_special_tokens=True,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                
                # 2. KG Features
                try:
                    stats = kg_builder.extract_features(input_text)
                    kg_features = torch.tensor(stats, dtype=torch.float).unsqueeze(0)
                except:
                    kg_features = torch.zeros(1, 3)
                    stats = [0, 0, 0]

                # 3. Predict
                with torch.no_grad():
                    input_ids = encoding['input_ids']
                    mask = encoding['attention_mask']
                    output = model(input_ids, mask, kg_features)
                    prob = torch.sigmoid(output).item()
                
                # Label mapping
                confidence = prob if prob > 0.5 else 1 - prob
                
                if confidence < 0.52:
                    label = "UNCERTAIN"
                elif confidence < 0.60:
                    label = "WEAK REAL" if prob > 0.5 else "WEAK FAKE"
                else:
                    label = "STRONG REAL" if prob > 0.5 else "STRONG FAKE"
                
                st.session_state['prediction'] = {
                    "label": label,
                    "score": confidence,
                    "kg_stats": stats
                }
                
                # 4. Build Graph
                G = kg_builder.build_graph(input_text)
                st.session_state['graph'] = G

                st.success("Analysis Complete!")

with col2:
    # --- Part 1: Prediction Result (Top Priority) ---
    st.subheader("📊 Detection Result")
    
    if 'prediction' in st.session_state:
        pred = st.session_state['prediction']
        score = pred['score']
        label = pred['label']
        stats = pred.get('kg_stats', [0,0,0])
        
        # Stylized Result Card
        if "UNCERTAIN" in label:
            st.warning(f"### ⚠️ {label}")
        elif "REAL" in label:
            st.success(f"### ✅ {label}")
        else:
            st.error(f"### 🛑 {label}")
            
        st.write(f"**Confidence:** {score:.1%}")
        st.progress(score)
        
        # Metrics Display
        c1, c2, c3 = st.columns(3)
        c1.metric("Nodes", int(stats[0]))
        c2.metric("Edges", int(stats[1]))
        c3.metric("Density", f"{stats[2]:.2f}")
        
    else:
        st.info("👈 Enter text and click 'Analyze' to see results.")

    st.markdown("---")

    # --- Part 2: Knowledge Graph (Below Prediction) ---
    st.subheader("🕸️ Knowledge Graph")
    
    if 'graph' in st.session_state:
        G = st.session_state['graph']
        
        if G.number_of_nodes() > 0:
            nodes = []
            edges = []
            
            for node_id in G.nodes:
                node_label = G.nodes[node_id].get('label', 'Entity')
                color = "#4CAF50" if node_label == "ORG" else "#2196F3" if node_label == "PERSON" else "#FF9800"
                nodes.append(Node(id=node_id, label=node_id, size=15, color=color))
            
            for u, v in G.edges:
                edge_label = G.edges[u, v].get('relation', '')
                edges.append(Edge(source=u, target=v, label=edge_label))
            
            config = Config(width=None, height=400, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
            
            with st.container(border=True):
                agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.caption("No specific entities relations found to visualize.")
    else:
        st.caption("Graph visualization will appear here after analysis.")