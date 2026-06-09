import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

from models.hierarchical import HierarchicalClustering

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="Hierarchical Clustering Dashboard",
    page_icon="🌳",
    layout="wide"
)

# ---------------------------------
# CSS
# ---------------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(
135deg,
#f8fafc,
#dbeafe
);
}

.main-title{
text-align:center;
font-size:48px;
font-weight:bold;
color:#2563eb;
}

.sub-title{
text-align:center;
font-size:18px;
color:#475569;
}

[data-testid="metric-container"]{
background:white;
border-radius:15px;
padding:15px;
border:2px solid #bfdbfe;
}

[data-testid="stSidebar"]{
background:#eff6ff;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# HEADER
# ---------------------------------

st.markdown("""
<div class='main-title'>
🌳 Hierarchical Clustering Dashboard
</div>
<div class='sub-title'>
Agglomerative Clustering on Iris Dataset
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------
# LOAD DATA
# ---------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.title("⚙ Settings")

clusters = st.sidebar.slider(
    "Number of Clusters",
    2,
    10,
    3
)

# ---------------------------------
# MODEL
# ---------------------------------

model = HierarchicalClustering(clusters)

result = model.fit(df)

X = result["X_scaled"]
labels = result["labels"]
score = result["score"]

df["Cluster"] = labels

# ---------------------------------
# KPI
# ---------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Flowers", len(df))
c2.metric("Features", 4)
c3.metric("Clusters", clusters)
c4.metric("Silhouette", round(score,3))

st.divider()

# ---------------------------------
# TABS
# ---------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
[
"Dataset",
"Dendrogram",
"Clusters",
"Export"
]
)

# ---------------------------------
# DATASET
# ---------------------------------

with tab1:

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(15),
        use_container_width=True
    )

    st.subheader("Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

# ---------------------------------
# DENDROGRAM
# ---------------------------------

with tab2:

    st.subheader("Hierarchical Dendrogram")

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    linked = linkage(
        X,
        method='ward'
    )

    dendrogram(
        linked,
        ax=ax
    )

    plt.title("Dendrogram")

    st.pyplot(fig)

# ---------------------------------
# CLUSTERS
# ---------------------------------

with tab3:

    fig = px.scatter(
        x=X[:,0],
        y=X[:,1],
        color=labels.astype(str),
        title="Cluster Visualization",
        color_discrete_sequence=
        px.colors.qualitative.Bold
    )

    fig.update_layout(
        template="plotly_white",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    counts = (
        df["Cluster"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Cluster",
        "Count"
    ]

    pie = px.pie(
        counts,
        names="Cluster",
        values="Count",
        hole=0.5
    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

# ---------------------------------
# EXPORT
# ---------------------------------

with tab4:

    csv = df.to_csv(index=False)

    st.download_button(
        "📥 Download Dataset",
        csv,
        "hierarchical_clusters.csv",
        "text/csv"
    )