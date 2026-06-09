import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_iris
from models.gmm import GMMClustering

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="GMM Dashboard",
    page_icon="📈",
    layout="wide"
)

# ----------------------------------
# CSS
# ----------------------------------

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

# ----------------------------------
# HEADER
# ----------------------------------

st.markdown("""
<div class='main-title'>
📈 Gaussian Mixture Model Dashboard
</div>

<div class='sub-title'>
Probabilistic Clustering using Iris Dataset
</div>
""", unsafe_allow_html=True)

st.divider()

# ----------------------------------
# LOAD DATASET
# ----------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("⚙ Settings")

components = st.sidebar.slider(
    "Number of Components",
    2,
    10,
    3
)

# ----------------------------------
# MODEL
# ----------------------------------

model = GMMClustering(components)

result = model.fit(df)

X = result["X_scaled"]
labels = result["labels"]
score = result["score"]
means = result["means"]
probabilities = result["probabilities"]

df["Cluster"] = labels

# ----------------------------------
# KPI SECTION
# ----------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Flowers", len(df))
c2.metric("Features", 4)
c3.metric("Components", components)
c4.metric("Silhouette", round(score,3))

st.divider()

# ----------------------------------
# TABS
# ----------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
[
"Dataset",
"Clusters",
"Analytics",
"Export"
]
)

# ----------------------------------
# DATASET
# ----------------------------------

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

# ----------------------------------
# CLUSTERS
# ----------------------------------

with tab2:

    fig = px.scatter(
        x=X[:,0],
        y=X[:,1],
        color=labels.astype(str),
        title="GMM Cluster Visualization",
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

    st.subheader("Gaussian Means")

    means_df = pd.DataFrame(
        means,
        columns=df.columns[:-1]
    )

    st.dataframe(
        means_df,
        use_container_width=True
    )

# ----------------------------------
# ANALYTICS
# ----------------------------------

with tab3:

    cluster_df = (
        df["Cluster"]
        .value_counts()
        .reset_index()
    )

    cluster_df.columns = [
        "Cluster",
        "Count"
    ]

    pie = px.pie(
        cluster_df,
        names="Cluster",
        values="Count",
        hole=0.5,
        title="Cluster Distribution"
    )

    pie.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

    st.subheader(
        "Cluster Membership Probabilities"
    )

    prob_df = pd.DataFrame(
        probabilities[:10]
    )

    st.dataframe(
        prob_df,
        use_container_width=True
    )

# ----------------------------------
# EXPORT
# ----------------------------------

with tab4:

    csv = df.to_csv(index=False)

    st.download_button(
        "📥 Download Clustered Dataset",
        csv,
        "gmm_clusters.csv",
        "text/csv"
    )