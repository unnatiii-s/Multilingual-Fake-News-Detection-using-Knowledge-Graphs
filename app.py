import os
import streamlit as st
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config

from graph import KnowledgeGraphBuilder
from model import FakeNewsModel

# --- Page Config ---
st.set_page_config(
    layout="wide",
    page_title="Multilingual Fake News Detection",
    page_icon="🕵️‍♂️"
)

# --- Custom CSS ---
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --- Load lightweight components ---
@st.cache_resource
def load_components():
    model = FakeNewsModel(use_kg=True)
    kg_builder = KnowledgeGraphBuilder()
    return model, kg_builder

model, kg_builder = load_components()

st.title("Multilingual Fake News Detection with Knowledge Graphs")

col1, col2 = st.columns([1, 1.5], gap="large")

# ================= LEFT =================
with col1:
    st.subheader("📝 Input News")

    input_text = st.text_area(
        "Enter news text here:",
        height=280,
        placeholder="e.g., The earth is flat and the moon is made of cheese...",
        key="news_input"
    )

    if st.button("🔍 Analyze"):
        if not input_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                prob = model.predict_proba(input_text)
                confidence = max(prob, 1 - prob)

                if confidence < 0.52:
                    label = "UNCERTAIN"
                elif confidence < 0.60:
                    label = "WEAK FAKE"
                else:
                    label = "STRONG FAKE"

                stats = kg_builder.extract_features(input_text)
                graph = kg_builder.build_graph(input_text)

                st.session_state["result"] = (label, confidence, stats, graph)
                st.success("Analysis complete!")

# ================= RIGHT =================
with col2:
    st.subheader("📊 Detection Result")

    if "result" in st.session_state:
        label, confidence, stats, G = st.session_state["result"]

        if label == "UNCERTAIN":
            st.warning(label)
        elif "WEAK" in label:
            st.error(label)
        else:
            st.error(label)

        st.write(f"**Confidence:** {confidence:.1%}")
        st.progress(confidence)

        c1, c2, c3 = st.columns(3)
        c1.metric("Nodes", stats[0])
        c2.metric("Edges", stats[1])
        c3.metric("Density", f"{stats[2]:.2f}")

        st.markdown("---")
        st.subheader("🕸️ Knowledge Graph")

        if G.number_of_nodes() > 0:
            nodes = [Node(id=n, label=n) for n in G.nodes]
            edges = []
            config = Config(height=350, directed=True)
            agraph(nodes, edges, config)
        else:
            st.info("No graph data available.")
