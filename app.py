import io
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from core import AutoDataCleaner, FeatureBuilder, ModelEngine, PipelinePredictor

st.set_page_config(page_title="Predictive OS", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Theme
# -----------------------------
st.markdown("""
<style>
.stApp { background:#f7f8fa; }
.block-container { max-width:1400px; padding:2rem 2.5rem 3rem; }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid #e5e7eb; }
h1,h2,h3,h4 { color:#111827; letter-spacing:-.02em; }
p, label, .stCaption { color:#667085; }
.metric-card { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px; min-height:118px; }
.metric-label { font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:#667085; }
.metric-value { font-size:26px; font-weight:700; color:#111827; margin-top:8px; }
.metric-help { font-size:12px; color:#98a2b3; margin-top:5px; }
.uvp { background:#111827; color:#fff; border-radius:14px; padding:20px 22px; margin:8px 0 24px; }
.uvp-title { font-size:11px; font-weight:700; letter-spacing:.10em; text-transform:uppercase; color:#aeb8cc; }
.uvp-main { font-size:22px; font-weight:700; margin:5px 0 7px; }
.uvp-copy { color:#d7dce5; font-size:13px; line-height:1.55; max-width:980px; }
.section { margin-top:22px; margin-bottom:10px; }
.small-note { font-size:12px; color:#98a2b3; }
.nav-label { font-size:11px; font-weight:700; color:#98a2b3; letter-spacing:.08em; text-transform:uppercase; margin:6px 0 8px; }
[data-testid="stSidebar"] .stButton > button { width:100%; justify-content:flex-start; text-align:left; border:1px solid transparent; background:transparent; color:#475467; border-radius:9px; padding:9px 12px; font-size:14px; font-weight:500; margin:2px 0; }
[data-testid="stSidebar"] .stButton > button:hover { background:#f2f4f7; border-color:#e4e7ec; color:#111827; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background:#eef2f7; color:#111827; border-color:#d9dee8; font-weight:650; }
.report-card { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:18px 20px; }
.status-pill { display:inline-block; padding:4px 8px; border-radius:999px; font-size:11px; font-weight:700; background:#ecfdf3; color:#027a48; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# State
# -----------------------------
def init_state():
    defaults = {
        "model_engine": None, "cleaner": None, "feature_builder": None,
        "results_dict": {}, "best_score": None, "X_cols": [],
        "drop_columns": [], "training_rows": 0, "training_features": 0,
        "training_time": None, "last_predictions": None, "audit_log": [],
        "confidence_threshold": 75, "drift_tolerance": 0.15,
        "slack_notifications": False, "email_summaries": False, "webhook_url": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

init_state()


def card(title, value, help_text):
    return f'''<div class="metric-card"><div class="metric-label">{title}</div><div class="metric-value">{value}</div><div class="metric-help">{help_text}</div></div>'''


def add_audit(event, details):
    st.session_state.audit_log.insert(0, {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Event": event,
        "Details": details,
    })
    st.session_state.audit_log = st.session_state.audit_log[:100]


def make_pdf_report():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 55
    c.setFont("Helvetica-Bold", 18); c.drawString(45, y, "Predictive OS — Governance Report")
    y -= 28
    c.setFont("Helvetica", 10)
    lines = [
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Best model: {st.session_state.model_engine.best_model_name if st.session_state.model_engine else 'N/A'}",
        f"Validation score: {st.session_state.best_score:.4f}" if st.session_state.best_score is not None else "Validation score: N/A",
        f"CV folds: {getattr(st.session_state.model_engine, 'cv_folds', 'N/A') if st.session_state.model_engine else 'N/A'}",
        f"Training rows: {st.session_state.training_rows}",
        f"Features: {st.session_state.training_features}",
        f"Human-audit threshold: {st.session_state.confidence_threshold}%",
        "",
        "Audit events:",
    ]
    for line in lines:
        c.drawString(45, y, line); y -= 16
    for event in st.session_state.audit_log[:20]:
        text = f"{event['Timestamp']} | {event['Event']} | {event['Details']}"
        c.drawString(55, y, text[:115]); y -= 14
        if y < 55:
            c.showPage(); y = height - 55; c.setFont("Helvetica", 10)
    c.save(); buf.seek(0)
    return buf.getvalue()

# -----------------------------
# Sidebar
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Overview"

def navigate(label):
    st.session_state.page = label

st.sidebar.markdown("<div style='padding:8px 0 20px'><div style='font-size:17px;font-weight:750;color:#111827;letter-spacing:-.02em'>Predictive OS</div><div style='font-size:11px;color:#98a2b3;margin-top:2px'>Industrial decision intelligence</div></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='nav-label'>Workspace</div>", unsafe_allow_html=True)
for label, key in [("Overview", "nav_overview"), ("Live Inference", "nav_live"), ("Intelligence Reports", "nav_reports"), ("System Settings", "nav_settings")]:
    if st.sidebar.button(label, key=key, use_container_width=True, type="primary" if st.session_state.page == label else "secondary"):
        navigate(label)
        st.rerun()
page = st.session_state.page

status = "Operational" if st.session_state.model_engine else "Awaiting model"
st.sidebar.markdown(f"<div style='position:fixed;bottom:22px;width:235px;background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:12px'><b style='font-size:12px'>SYSTEM</b><br><span style='font-size:12px;color:{'#12b76a' if status == 'Operational' else '#f79009'}'>● {status}</span><br><span style='font-size:11px;color:#98a2b3'>Predictive OS v1.1</span></div>", unsafe_allow_html=True)

# -----------------------------
# Overview
# -----------------------------
if page == "Overview":
    st.title("Predictive OS")
    st.caption("Decision infrastructure for industrial telemetry — from raw sensor data to governed maintenance decisions.")
    st.markdown("""
    <div class="uvp"><div class="uvp-title">Decision intelligence for industrial operations</div>
    <div class="uvp-main">Predict with confidence. Escalate with control.</div>
    <div class="uvp-copy">Predictive OS benchmarks multiple ML models, scores unseen telemetry, and applies a confidence gate before an automated decision is accepted. Uncertain cases are routed to human review and recorded in an auditable decision trail.</div></div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    if st.session_state.model_engine:
        c1.markdown(card("Status", "READY", "Pipeline trained and available"), unsafe_allow_html=True)
        c2.markdown(card("Best Model", st.session_state.model_engine.best_model_name, "Selected by cross-validation"), unsafe_allow_html=True)
        c3.markdown(card("Validation", f"{st.session_state.best_score:.3f}", "F1 macro for classification"), unsafe_allow_html=True)
        c4.markdown(card("Features", st.session_state.training_features, "Stable inference schema"), unsafe_allow_html=True)
    else:
        c1.markdown(card("Status", "SETUP", "Upload telemetry to begin"), unsafe_allow_html=True)
        c2.markdown(card("Models", "3", "RF · XGBoost · LightGBM"), unsafe_allow_html=True)
        c3.markdown(card("Validation", "5-fold", "Adaptive when classes are small"), unsafe_allow_html=True)
        c4.markdown(card("Audit", "READY", "Confidence-gated review"), unsafe_allow_html=True)

    st.markdown("<div class='section'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.55, 1])
    with left:
        st.subheader("Train a decision pipeline")
        uploaded = st.file_uploader("Training telemetry CSV", type="csv", key="train_upload")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                df.columns = df.columns.str.strip()
                if df.empty:
                    st.error("The CSV is empty.")
                else:
                    default_idx = list(df.columns).index("Machine failure") if "Machine failure" in df.columns else len(df.columns)-1
                    a,b = st.columns(2)
                    with a:
                        target = st.selectbox("Target variable", df.columns, index=default_idx)
                        task = st.radio("Task", ["classification", "regression"], horizontal=True)
                    with b:
                        drops = st.multiselect("Optional columns to exclude", [c for c in df.columns if c != target])
                        st.caption(f"{len(df):,} rows · {len(df.columns):,} columns")

                    if st.button("Train & benchmark", type="primary", use_container_width=True):
                        if target in drops:
                            st.error("The target cannot also be excluded.")
                        else:
                            with st.spinner("Cleaning telemetry, benchmarking models, and fitting the winner…"):
                                cleaner = AutoDataCleaner(df)
                                cleaned = cleaner.clean_all()
                                if target not in cleaned.columns:
                                    raise ValueError("The selected target was removed during cleaning. Check missing values in the target column.")
                                y = cleaned[target]
                                X_raw = cleaned.drop(columns=[target] + drops, errors="ignore")
                                if X_raw.shape[1] == 0:
                                    raise ValueError("No predictor features remain after exclusions.")
                                builder = FeatureBuilder()
                                X = builder.fit_transform(X_raw)
                                engine = ModelEngine(task_type=task)
                                results, best = engine.train_and_benchmark(X, y)

                                st.session_state.update({
                                    "cleaner": cleaner, "feature_builder": builder, "model_engine": engine,
                                    "results_dict": results, "best_score": results[best],
                                    "X_cols": list(X.columns), "drop_columns": drops,
                                    "training_rows": len(cleaned), "training_features": X.shape[1],
                                    "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "last_predictions": None,
                                })
                                add_audit("MODEL_TRAINED", f"{best} selected from RF/XGBoost/LightGBM")
                                st.success(f"Pipeline ready. {best} selected with validation score {results[best]:.4f}.")
                                st.rerun()
            except Exception as exc:
                st.error(f"Training failed safely: {exc}")
    with right:
        st.subheader("Model intelligence")
        if st.session_state.model_engine:
            scores = pd.DataFrame({"Model": list(st.session_state.results_dict.keys()), "Score": list(st.session_state.results_dict.values())})
            scores = scores.sort_values("Score", ascending=False).reset_index(drop=True)
            scores["Score"] = scores["Score"].map(lambda x: f"{x:.3f}")
            st.dataframe(scores, use_container_width=True, hide_index=True)
            model = st.session_state.model_engine.best_model
            if hasattr(model, "feature_importances_"):
                imp = pd.DataFrame({"Feature": st.session_state.X_cols, "Importance": model.feature_importances_}).nlargest(7, "Importance").sort_values("Importance")
                fig = px.bar(imp, x="Importance", y="Feature", orientation="h")
                fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train a pipeline to populate model comparison and feature impact.")

# -----------------------------
# Live inference
# -----------------------------
elif page == "Live Inference":
    st.title("Live Inference")
    st.caption("Run unseen telemetry through the exact training-time cleaning and feature schema.")
    if not st.session_state.model_engine:
        st.warning("Train a pipeline from Overview first.")
    else:
        f = st.file_uploader("Unseen telemetry CSV", type="csv", key="infer_upload")
        if f:
            try:
                test_df = pd.read_csv(f)
                test_df.columns = test_df.columns.str.strip()
                st.caption(f"Loaded {len(test_df):,} records")
                if st.button("Generate decisions", type="primary", use_container_width=True):
                    with st.spinner("Scoring telemetry…"):
                        predictor = PipelinePredictor(st.session_state.cleaner, st.session_state.feature_builder, st.session_state.model_engine)
                        test_clean = test_df.drop(columns=st.session_state.drop_columns, errors="ignore")
                        predictions = predictor.predict_unseen(test_clean, st.session_state.confidence_threshold)
                        st.session_state.last_predictions = predictions
                        review_count = int(predictions["Requires Human Audit"].sum()) if "Requires Human Audit" in predictions else 0
                        add_audit("INFERENCE_RUN", f"{len(predictions)} records scored; {review_count} routed to human review")
            except Exception as exc:
                st.error(f"Inference failed safely: {exc}")

        if st.session_state.last_predictions is not None:
            pred = st.session_state.last_predictions
            m1,m2,m3 = st.columns(3)
            m1.markdown(card("Records scored", f"{len(pred):,}", "Unseen telemetry"), unsafe_allow_html=True)
            if "Requires Human Audit" in pred:
                review = int(pred["Requires Human Audit"].sum())
                m2.markdown(card("Human review", f"{review:,}", f"Below {st.session_state.confidence_threshold}% confidence"), unsafe_allow_html=True)
                m3.markdown(card("Auto decisions", f"{len(pred)-review:,}", "Above configured threshold"), unsafe_allow_html=True)
            st.markdown("<div class='section'></div>", unsafe_allow_html=True)
            st.dataframe(pred.head(250), use_container_width=True, hide_index=True)
            st.download_button("Download decision CSV", pred.to_csv(index=False), "predictive_os_decisions.csv", "text/csv", type="primary")

# -----------------------------
# Intelligence Reports
# -----------------------------
elif page == "Intelligence Reports":
    st.title("Intelligence Reports")
    st.caption("A concise operational record of model performance, decision controls, and recent activity.")
    if not st.session_state.model_engine:
        st.info("Train a pipeline from Overview to generate intelligence and governance records.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(card("Production candidate", st.session_state.model_engine.best_model_name, "Highest cross-validation score"), unsafe_allow_html=True)
        c2.markdown(card("Validation score", f"{st.session_state.best_score:.3f}", "Macro F1" if st.session_state.model_engine.task_type == "classification" else "R²"), unsafe_allow_html=True)
        c3.markdown(card("Records trained", f"{st.session_state.training_rows:,}", "Telemetry rows used"), unsafe_allow_html=True)
        c4.markdown(card("Review threshold", f"{st.session_state.confidence_threshold}%", "Human-review gate"), unsafe_allow_html=True)

        st.markdown("<div class='section'></div>", unsafe_allow_html=True)
        left, right = st.columns([1.25, 1])
        with left:
            st.subheader("Model benchmark")
            scores = pd.DataFrame({"Model": list(st.session_state.results_dict.keys()), "Score": list(st.session_state.results_dict.values())})
            scores = scores.sort_values("Score", ascending=False).reset_index(drop=True)
            scores["Score"] = scores["Score"].round(4)
            scores["Selection"] = scores["Model"].eq(st.session_state.model_engine.best_model_name).map({True: "Selected", False: "Evaluated"})
            st.dataframe(scores, use_container_width=True, hide_index=True)
            st.caption(f"Selection based on {st.session_state.model_engine.cv_folds}-fold cross-validation.")
        with right:
            st.subheader("Governance posture")
            st.markdown(
                f"<div class='report-card'><div style='font-size:12px;color:#667085'>Decision gate</div>"
                f"<div style='font-size:22px;font-weight:700;color:#111827;margin:4px 0 12px'>Confidence ≥ {st.session_state.confidence_threshold}%</div>"
                f"<div style='font-size:13px;color:#667085;line-height:1.55'>Predictions below the configured threshold are marked for human review instead of being treated as automatic decisions.</div>"
                f"<div style='margin-top:14px'><span class='status-pill'>AUDIT ENABLED</span></div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='section'></div>", unsafe_allow_html=True)
        st.subheader("Decision activity")
        logs = pd.DataFrame(st.session_state.audit_log)
        if logs.empty:
            st.info("No governance events recorded yet. Run inference or update a setting to create the first event.")
        else:
            st.dataframe(logs, use_container_width=True, hide_index=True)

        st.markdown("<div class='section'></div>", unsafe_allow_html=True)
        b1,b2 = st.columns(2)
        with b1:
            st.download_button("Export governance report", make_pdf_report(), "predictive_os_governance.pdf", "application/pdf", type="primary", use_container_width=True)
        with b2:
            if st.session_state.last_predictions is not None:
                st.download_button("Export latest decisions", st.session_state.last_predictions.to_csv(index=False), "predictive_os_decisions.csv", "text/csv", use_container_width=True)
            else:
                st.button("Export latest decisions", disabled=True, use_container_width=True)

# -----------------------------
# Settings
# -----------------------------
else:
    st.title("System Settings")
    st.caption("Tune governance behavior without changing the trained model.")
    a,b = st.columns(2)
    with a:
        st.subheader("Decision governance")
        threshold = st.slider("Human-review confidence threshold", 50, 99, st.session_state.confidence_threshold)
        drift = st.slider("Data-drift tolerance", 0.0, 1.0, st.session_state.drift_tolerance, 0.01)
        st.info("Predictions below the confidence threshold are routed to human review. This is the Predictive OS safety gate.")
    with b:
        st.subheader("Integrations")
        slack = st.toggle("Slack alerts", st.session_state.slack_notifications)
        email = st.toggle("Email audit summaries", st.session_state.email_summaries)
        webhook = st.text_input("Webhook URL", st.session_state.webhook_url)
    if st.button("Save configuration", type="primary"):
        st.session_state.update({"confidence_threshold": threshold, "drift_tolerance": drift, "slack_notifications": slack, "email_summaries": email, "webhook_url": webhook})
        add_audit("SETTINGS_UPDATED", f"Human-review threshold set to {threshold}%")
        st.success("Configuration saved for this session.")
