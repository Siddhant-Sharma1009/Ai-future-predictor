import os
import json
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

# 🔥 NEW: Import Rule-Based Engine
from ml_engine import calculate_success_score, recommend_career, industry_comparison

# ==================================================
# LOAD TRAINED MODEL
# ==================================================

with open("career_model.pkl", "rb") as f:
    ml_model = pickle.load(f)

# ==================================================
# GEMINI SETUP
# ==================================================

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)

# ==================================================
# RESPONSE SCHEMA
# ==================================================

class FuturePrediction(BaseModel):
    career_5_year: str
    career_10_year: str
    income_projection: str
    skill_gaps: str
    risks: str
    future_letter: str

    openness: int = Field(ge=0, le=100)
    conscientiousness: int = Field(ge=0, le=100)
    extraversion: int = Field(ge=0, le=100)
    agreeableness: int = Field(ge=0, le=100)
    emotional_stability: int = Field(ge=0, le=100)


# ==================================================
# SAFE JSON PARSER
# ==================================================

def safe_parse_response(response):
    if hasattr(response, "parsed") and response.parsed:
        return response.parsed.model_dump()

    try:
        return json.loads(response.text)
    except Exception:
        return {"error": response.text}


# ==================================================
# MAIN HYBRID FUNCTION
# ==================================================

def generate_future_prediction(age, field, daily_hours, skills, habits, goals):

    # -------------------------------
    # GEMINI PART
    # -------------------------------

    prompt = f"""
You are an advanced AI career forecasting system.

USER PROFILE:
Age: {age}
Field: {field}
Daily Productive Hours: {daily_hours}
Skills: {skills}
Habits: {habits}
Long-Term Goals: {goals}

Return ONLY valid JSON matching schema.
Do NOT add markdown.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": FuturePrediction,
                "temperature": 0.7,
            },
        )

        gemini_result = safe_parse_response(response)

        if "error" in gemini_result:
            return gemini_result

    except Exception as e:
        return {"error": str(e)}

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------

    skill_count = len([s.strip() for s in skills.split(",") if s.strip()])
    habit_strength = len(habits.split())

    X = pd.DataFrame([{
        "age": age,
        "daily_hours": daily_hours,
        "skill_count": skill_count,
        "habit_strength": habit_strength,
        "openness": gemini_result["openness"],
        "conscientiousness": gemini_result["conscientiousness"],
        "extraversion": gemini_result["extraversion"],
        "agreeableness": gemini_result["agreeableness"],
        "emotional_stability": gemini_result["emotional_stability"],
        "field": field
    }])

    # -------------------------------
    # RANDOM FOREST PREDICTION
    # -------------------------------

    prediction = ml_model.predict(X)[0]

    # Confidence estimation
    preprocessed_X = ml_model.named_steps["preprocessor"].transform(X)
    trees = ml_model.named_steps["model"].estimators_

    tree_predictions = np.array(
        [tree.predict(preprocessed_X)[0] for tree in trees]
    )

    confidence = int(100 - np.std(tree_predictions))
    confidence = max(50, min(confidence, 95))

    # Feature importance
    feature_names = ml_model.named_steps["preprocessor"].get_feature_names_out()
    importances = ml_model.named_steps["model"].feature_importances_
    feature_importance = dict(zip(feature_names, importances))

    # -------------------------------
    # RULE-BASED ENGINE
    # -------------------------------

    personality_scores = {
        "openness": gemini_result["openness"],
        "conscientiousness": gemini_result["conscientiousness"],
        "extraversion": gemini_result["extraversion"],
        "agreeableness": gemini_result["agreeableness"],
        "emotional_stability": gemini_result["emotional_stability"],
    }

    rule_score, breakdown = calculate_success_score(
        daily_hours, skills, habits, personality_scores
    )

    recommended_role = recommend_career(personality_scores)
    industry_fit = industry_comparison(personality_scores)

    # -------------------------------
    # TRUE HYBRID SCORE
    # -------------------------------

    hybrid_score = int((prediction * 0.65) + (rule_score * 0.35))

    # -------------------------------
    # CAREER PATH ROADMAP
    # -------------------------------

    career_path = [
        f"Year 1-2: Strengthen core {field} fundamentals and build strong portfolio projects.",
        "Year 3-4: Take leadership in projects, collaborate, and improve communication skills.",
        f"Year 5+: Transition into {recommended_role} or a specialized advanced role."
    ]

    # -------------------------------
    # MERGE EVERYTHING
    # -------------------------------

    gemini_result["success_probability"] = hybrid_score
    gemini_result["rf_score"] = int(prediction)
    gemini_result["rule_score"] = rule_score
    gemini_result["score_breakdown"] = breakdown
    gemini_result["prediction_confidence"] = confidence
    gemini_result["feature_importance"] = feature_importance
    gemini_result["recommended_role"] = recommended_role
    gemini_result["industry_fit"] = industry_fit
    gemini_result["career_path"] = career_path

    return gemini_result