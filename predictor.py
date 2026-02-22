import os
import json
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from ml_engine import calculate_success_score, recommend_career, industry_comparison

# LOAD TRAINED MODEL

with open("career_model.pkl", "rb") as f:
    ml_model = pickle.load(f)

# GEMINI SETUP

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

genai.configure(api_key=api_key)

def safe_parse_response(text):
    try:
        return json.loads(text)
    except Exception:
        return {"error": text}


def generate_future_prediction(age, field, daily_hours, skills, habits, goals):

    prompt = f"""
You are an advanced AI career forecasting system.

USER PROFILE:
Age: {age}
Field: {field}
Daily Productive Hours: {daily_hours}
Skills: {skills}
Habits: {habits}
Long-Term Goals: {goals}

Return ONLY valid JSON in this format:

{{
"career_5_year": "",
"career_10_year": "",
"income_projection": "",
"skill_gaps": "",
"risks": "",
"future_letter": "",
"openness": 0,
"conscientiousness": 0,
"extraversion": 0,
"agreeableness": 0,
"emotional_stability": 0
}}

Do NOT include markdown.
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
            }
        )

        gemini_result = safe_parse_response(response.text)

        if "error" in gemini_result:
            return gemini_result

    except Exception as e:
        return {"error": str(e)}


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


    prediction = ml_model.predict(X)[0]

    
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

    hybrid_score = int((prediction * 0.65) + (rule_score * 0.35))

    career_path = [
        f"Year 1-2: Strengthen core {field} fundamentals and build strong portfolio projects.",
        "Year 3-4: Take leadership in projects, collaborate, and improve communication skills.",
        f"Year 5+: Transition into {recommended_role} or a specialized advanced role."
    ]

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
