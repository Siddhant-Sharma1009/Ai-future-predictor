import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io
from predictor import generate_future_prediction
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# PDF GENERATION

def create_pdf(result):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("AI Future Self Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.4 * inch))

    for key, value in result.items():
        if key not in ["feature_importance"]:
            elements.append(Paragraph(f"<b>{key}</b>", styles["Normal"]))
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph(str(value), styles["Normal"]))
            elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer


st.set_page_config(page_title="AI Future Predictor", page_icon="🔮", layout="wide")

st.title("🎲 AI Future Predictor")
st.caption("Hybrid AI System (Gemini + RandomForest + Rule-Based Engine)")
st.markdown("---")

with st.form("future_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=15, max_value=60)
        field = st.selectbox("Field", ["Technology", "Business", "Research", "Creative","Other"])
        daily_hours = st.slider("Daily Productive Hours", 0, 12)

    with col2:
        skills = st.text_area("Current Skills (comma separated)")
        habits = st.text_area("Habits")
        goals = st.text_area("Long-term Goals")

    submitted = st.form_submit_button("🚀 Generate Prediction")


if submitted:

    with st.spinner("Analyzing your future trajectory..."):
        result = generate_future_prediction(age, field, daily_hours, skills, habits, goals)

    if "error" in result:
        st.error(result["error"])
    else:

        st.subheader("🚀 Career Projection")

        colA, colB = st.columns(2)
        colA.info(result["career_5_year"])
        colB.success(result["career_10_year"])

        st.markdown("### 💰 Income Outlook")
        st.write(result["income_projection"])

        st.markdown("### 📉 Skill Gaps")
        st.warning(result["skill_gaps"])

        st.markdown("### ⚠ Potential Risks")
        st.error(result["risks"])

        # HYBRID SUCCESS FORECAST

        st.markdown("---")
        st.subheader("🎯 Hybrid Success Forecast")

        st.progress(result["success_probability"])
        st.markdown(f"### Final Hybrid Score: {result['success_probability']}%")

        col1, col2, col3 = st.columns(3)
        col1.metric("Random Forest Score", f"{result['rf_score']}%")
        col2.metric("Rule-Based Score", f"{result['rule_score']}%")
        col3.metric("AI Confidence", f"{result['prediction_confidence']}%")

        st.markdown("#### Score Breakdown")
        st.json(result["score_breakdown"])

        # Growth Graph 

        st.markdown("---")
        st.subheader("📈 10-Year Growth Projection")

        years = np.arange(0, 11)
        growth = result["success_probability"] + (100 - result["success_probability"]) * (1 - np.exp(-0.25 * years))

        fig, ax = plt.subplots(figsize=(5,2.5))
        ax.plot(years, growth)
        ax.set_ylim(0,100)
        ax.set_xlabel("Years")
        ax.set_ylabel("Career Index")
        ax.grid(alpha=0.2)

        st.pyplot(fig)

        # Career Path

        st.markdown("---")
        st.subheader("🛤 Suggested Career Path")

        for step in result["career_path"]:
            st.write("• " + step)

        st.markdown(f"### 🎯 Recommended Role: {result['recommended_role']}")
        st.markdown("### 🏢 Industry Compatibility")
        st.json(result["industry_fit"])

        # Feature Importance 

        st.markdown("---")
        st.subheader("🔍 ML Feature Importance")

        feature_importance = result["feature_importance"]
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:8]

        labels = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]

        fig2, ax2 = plt.subplots(figsize=(5,3))
        ax2.barh(labels[::-1], values[::-1])
        ax2.set_xlabel("Importance Score")

        st.pyplot(fig2)

        # Future Letter

        st.markdown("---")
        st.subheader("📜 Letter from Your Future Self")
        st.write(result["future_letter"])

        # PDF Download

        pdf_file = create_pdf(result)

        st.download_button(
            "📄 Download Full Report (PDF)",
            data=pdf_file,
            file_name="AI_Future_Report.pdf",
            mime="application/pdf"
        )
