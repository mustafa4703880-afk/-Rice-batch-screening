import json
import sqlite3
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from history import read_batches, read_report, save_batch
from screening import read_measurements, screen_grains
from train_model import FEATURES, PROFILES

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Rice Batch Screening", page_icon="🌾", layout="wide")


@st.cache_resource
def load_models():
    return joblib.load(BASE / "models" / "rice_profiles.joblib")


st.markdown("""
<style>
    .block-container {padding-top: 2.6rem; max-width: 1250px;}
    [data-testid="stMetric"] {
        border:1px solid #D7E5D9; background:#F5F8F3; border-radius:12px; padding:14px;
    }
    .rice-intro {
        padding: 26px 30px; margin-bottom: 20px; border-radius: 16px;
        background: #153D30; color: #F9F9EA;
    }
    .rice-intro h1 {margin:0; padding:0; color:#F9F9EA; font-size:2.4rem;}
    .rice-intro p {margin:8px 0 0; color:#D7E7D9;}
</style>
<div class="rice-intro">
    <h1>Rice Batch Screening</h1>
    <p>Upload grain measurements, review variety matches, and save the lot report.</p>
</div>
""", unsafe_allow_html=True)

models = load_models()
metrics = json.loads((BASE / "models" / "metrics.json").read_text(encoding="utf-8"))
profile_choices = ["Basmati / Jasmine / Karacadag", "Cammeo / Osmancik"]
screen_tab, history_tab, single_tab, model_tab = st.tabs(
    ["Screen Batch", "Batch History", "Single Grain", "Model Results"]
)

with screen_tab:
    st.subheader("New batch")
    left, right = st.columns([3, 2], gap="large")
    with left:
        profile_name = st.selectbox("Measurement profile", profile_choices)
        profile = models[profile_name]
        classes = profile["model"].classes_.tolist()
        declared = st.selectbox("Declared variety", classes, key=f"declared_{profile_name}")
        st.caption("Choose the profile that matches your measurement method and units.")
    with right:
        sample_number = list(PROFILES).index(profile_name) + 1
        demo = (BASE / "examples" / f"demo_{sample_number}.csv").read_bytes()
        st.download_button("Download sample CSV", demo, "rice_sample.csv", "text/csv")
        st.download_button("Download blank template", ",".join(FEATURES) + "\n",
                           "rice_template.csv", "text/csv")
        demo_mode = st.checkbox("Use built-in sample")
    upload = st.file_uploader("Upload grain measurements", type=["csv"], disabled=demo_mode)
    with st.form("screen_batch"):
        details = st.columns(3)
        lot = details[0].text_input("Lot ID", placeholder="LOT-024")
        supplier = details[1].text_input("Supplier", placeholder="Supplier name")
        batch_date = details[2].date_input("Batch date", value=date.today())
        with st.expander("CSV measurements and review threshold"):
            st.write("One row per grain. Use the same measurement method and units as the selected profile.")
            st.code(", ".join(FEATURES), language=None)
            threshold = st.slider("Minimum model score (%)", 50, 95, 70)
            st.caption("Pilot setting. Scores below this value are flagged for operator review.")
        run_batch = st.form_submit_button("Screen batch", type="primary")
    if run_batch:
        st.session_state.pop("batch", None)
        if not (demo_mode or upload):
            st.error("Upload a CSV or select Use built-in sample.")
        elif not demo_mode and (not lot.strip() or not supplier.strip()):
            st.error("Enter a Lot ID and Supplier for an uploaded batch.")
        else:
            try:
                grains = read_measurements(demo if demo_mode else upload.getvalue())
            except ValueError as error:
                st.error(str(error))
            else:
                result = screen_grains(grains, profile, declared, threshold / 100)
                status = "Needs review" if result["Screening result"].ne("Declared match").any() else "No flags"
                st.session_state.batch = {
                    "result": result, "lot": lot.strip() or "DEMO-LOT",
                    "supplier": supplier.strip() or "Demo sample", "date": batch_date,
                    "declared": declared, "profile": profile_name, "threshold": threshold,
                    "status": status,
                }
                st.session_state.pop("qa_decision", None)
                st.session_state.pop("qa_notes", None)

    if batch := st.session_state.get("batch"):
        result = batch["result"]
        counts = result["Screening result"].value_counts()
        valid = len(result) - counts.get("Invalid input", 0)
        st.divider()
        title, status_panel = st.columns([3, 1])
        title.subheader(f"Results · {batch['lot']}")
        title.text(f"{batch['supplier']}  ·  {batch['date']}  ·  Declared {batch['declared']}")
        (status_panel.warning if batch["status"] == "Needs review" else status_panel.success)(batch["status"])
        for column, (name, value) in zip(st.columns(5), [
            ("Valid grains", f"{valid:,}"),
            ("Declared match", f"{counts.get('Declared match', 0):,}"),
            ("Different variety", f"{counts.get('Different variety', 0):,}"),
            ("Uncertain", f"{counts.get('Uncertain', 0):,}"),
            ("Input issues", f"{counts.get('Invalid input', 0):,}"),
        ]):
            column.metric(name, value)
        distribution = result.loc[result["Screening result"].isin(
            ["Declared match", "Different variety"]), "Predicted variety"].value_counts()
        if not distribution.empty:
            st.subheader("Predicted varieties")
            st.caption("Counts include classified rows that cleared the review threshold and range check.")
            st.bar_chart(distribution.rename("Grains"), horizontal=True, color="#26765D")
        st.subheader("Grain results")
        view = st.radio("Rows to show", ["Flagged rows", "All rows"], horizontal=True)
        shown = result.loc[result["Screening result"].ne("Declared match")] if view == "Flagged rows" else result
        columns = ["Grain row", "Screening result", "Predicted variety", "Model score (%)", "Reason"]
        if shown.empty:
            st.info("No flagged rows.")
        else:
            st.dataframe(shown[columns].head(1000), hide_index=True, width="stretch")
            if len(shown) > 1000:
                st.caption("Showing the first 1,000 matching rows. The download contains every row.")
        def report_csv(decision="", notes=""):
            export = result[columns + FEATURES].assign(**{
                "Lot ID": batch["lot"], "Supplier": batch["supplier"],
                "Batch date": str(batch["date"]), "Declared variety": batch["declared"],
                "Measurement profile": batch["profile"], "Minimum score (%)": batch["threshold"],
                "Screening status": batch["status"], "Operator decision": decision, "Review notes": notes,
            })
            for field in ("Lot ID", "Supplier", "Review notes"):
                export[field] = export[field].map(
                    lambda value: "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value
                )
            return export.to_csv(index=False)

        st.download_button("Download full results", report_csv(), "rice_batch_results.csv", "text/csv")
        st.subheader("QA review")
        with st.form("qa_review"):
            decision = st.selectbox("Operator decision", ["Pending", "Accepted", "Hold", "Rejected"], key="qa_decision")
            notes = st.text_area("Review notes", key="qa_notes")
            if st.form_submit_button("Save to batch history", type="primary"):
                try:
                    save_batch(batch, decision, notes, report_csv(decision, notes))
                except sqlite3.IntegrityError:
                    st.error("A review for this Lot ID, Supplier and Date is already saved.")
                else:
                    st.success("Batch saved.")

with history_tab:
    st.subheader("Saved batches")
    saved = read_batches()
    if not saved:
        st.info("Saved QA reviews will appear here.")
    else:
        query = st.text_input("Find a batch", placeholder="Search by lot, supplier or date").strip().casefold()
        rows = [row for row in saved if query in f"{row['lot']} {row['supplier']} {row['batch_date']}".casefold()]
        if rows:
            summary = [{
                "Lot ID": row["lot"], "Supplier": row["supplier"], "Date": row["batch_date"],
                "Declared variety": row["variety"], "Screening status": row["status"],
                "Operator decision": row["decision"],
            } for row in rows]
            st.dataframe(summary, hide_index=True, width="stretch")
            by_id = {row["id"]: row for row in rows}
            selected = st.selectbox("Open saved report", list(by_id),
                                    format_func=lambda i: f"{by_id[i]['lot']} · {by_id[i]['supplier']} · {by_id[i]['batch_date']}")
            report = by_id[selected]
            st.caption(f"Measurement profile: {report['profile']} · Minimum model score: {report['threshold']}%")
            if report["notes"]:
                st.text(report["notes"])
            st.download_button("Download saved report", read_report(selected), "saved_rice_report.csv", "text/csv")
        else:
            st.info("No batches match your search.")

with single_tab:
    st.subheader("Check one grain")
    name = st.selectbox("Measurement profile", profile_choices, key="single_profile")
    single = models[name]
    example = st.selectbox("Start from example", single["examples"].index.tolist())
    fields = {}
    panes = st.columns(3)
    for index, feature in enumerate(FEATURES):
        with panes[index % 3]:
            fields[feature] = st.number_input(
                feature.replace("_", " "), value=float(single["examples"].loc[example, feature]),
                format="%.5f", key=f"{name}-{example}-{feature}",
            )
    screened = screen_grains(pd.DataFrame([fields]), single, example, 0.70).iloc[0]
    if screened["Screening result"] == "Invalid input":
        st.error(f"Check these measurements: {screened['Reason']}.")
    else:
        prediction, score = st.columns(2)
        prediction.metric("Predicted variety", screened["Predicted variety"])
        score.metric("Model score", f"{screened['Model score (%)']:.1f}%")
        if screened["Screening result"] == "Uncertain":
            st.warning(screened["Reason"])

with model_tab:
    st.subheader("Validation results")
    name = st.selectbox("Measurement profile", profile_choices, key="results_profile")
    report = metrics[name]
    for column, (key, value) in zip(st.columns(3), [
        ("Original records", f"{report['source_rows']:,}"),
        ("Test accuracy", f"{report['accuracy']:.2%}"),
        ("Macro F1", f"{report['macro_f1']:.2%}"),
    ]):
        column.metric(key, value)
    st.caption(f"Random Forest · {report['training_rows']:,} training rows · {report['test_rows']:,} held-out rows from the research dataset")
    st.dataframe(pd.DataFrame(
        report["confusion"], index=[f"Actual {label}" for label in report["classes"]],
        columns=[f"Predicted {label}" for label in report["classes"]],
    ), width="stretch")
    st.info("Scope: uploaded samples measured using the selected profile. Validate on independent supplier lots before live QA use.")
    st.markdown("[Rice MSC research dataset](https://www.muratkoklu.com/datasets/) · "
                "[UCI Cammeo / Osmancik dataset](https://doi.org/10.24432/C5MW4Z)")
