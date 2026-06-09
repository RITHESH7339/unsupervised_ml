import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_iris

from models.isolation_forest import IsolationForestModel

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="Isolation Forest Dashboard",
    page_icon="🚨",
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
color:#dc2626;
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
border:2px solid #fecaca;
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
🚨 Isolation Forest Dashboard
</div>

<div class='sub-title'>
Anomaly Detection using Isolation Forest
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

contamination = st.sidebar.slider(
    "Contamination",
    min_value=0.01,
    max_value=0.30,
    value=0.05,
    step=0.01
)

# ----------------------------------
# MODEL
# ----------------------------------

model = IsolationForestModel(
    contamination=contamination
)

result = model.fit(df)

X = result["X_scaled"]
predictions = result["predictions"]
scores = result["scores"]

df["Anomaly"] = predictions

# ----------------------------------
# KPI
# ----------------------------------

normal_count = sum(predictions == 1)
anomaly_count = sum(predictions == -1)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Samples", len(df))
c2.metric("Features", 4)
c3.metric("Normal Points", normal_count)
c4.metric("Anomalies", anomaly_count)

st.divider()

# ----------------------------------
# TABS
# ----------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
[
"Dataset",
"Anomaly Detection",
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
        df.head(),
        use_container_width=True
    )

    st.subheader("Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

# ----------------------------------
# VISUALIZATION
# ----------------------------------

with tab2:

    plot_df = pd.DataFrame({
        "Feature1": X[:,0],
        "Feature2": X[:,1],
        "Anomaly": predictions.astype(str)
    })

    fig = px.scatter(
        plot_df,
        x="Feature1",
        y="Feature2",
        color="Anomaly",
        title="Anomaly Detection",
        color_discrete_map={
            "1":"blue",
            "-1":"red"
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ----------------------------------
# ANALYTICS
# ----------------------------------

with tab3:

    score_df = pd.DataFrame({
        "Anomaly Score": scores
    })

    hist = px.histogram(
        score_df,
        x="Anomaly Score",
        nbins=30,
        title="Anomaly Score Distribution"
    )

    hist.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(
        hist,
        use_container_width=True
    )

    st.subheader(
        "Detected Anomalies"
    )

    anomalies = df[
        df["Anomaly"] == -1
    ]

    st.dataframe(
        anomalies,
        use_container_width=True
    )

# ----------------------------------
# EXPORT
# ----------------------------------

with tab4:

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Download Results",
        csv,
        "isolation_forest_results.csv",
        "text/csv"
    )