import numpy as np

# RULE-BASED SUCCESS SCORING ENGINE

def calculate_success_score(daily_hours, skills, habits, personality_scores):

    skill_list = [s.strip() for s in skills.split(",") if s.strip()]
    habit_list = habits.split()

    hours_score = daily_hours / 12
    skill_score = min(len(skill_list) / 20, 1)
    habit_score = min(len(habit_list) / 150, 1)

    personality_score = (
        personality_scores["conscientiousness"] * 0.3 +
        personality_scores["emotional_stability"] * 0.25 +
        personality_scores["openness"] * 0.2 +
        personality_scores["extraversion"] * 0.15 +
        personality_scores["agreeableness"] * 0.1
    ) / 100

    effort_component = (
        (hours_score * 0.4) +
        (skill_score * 0.35) +
        (habit_score * 0.25)
    )

    final_score = (effort_component * 70) + (personality_score * 30)

    return int(final_score), {
        "Effort Score": int(effort_component * 70),
        "Personality Score": int(personality_score * 30),
    }

# CAREER RECOMMENDATION ENGINE

def recommend_career(personality_scores):

    if personality_scores["openness"] > 75 and personality_scores["conscientiousness"] > 70:
        return "AI Researcher or Data Scientist"

    if personality_scores["extraversion"] > 70:
        return "Product Manager or Entrepreneur"

    if personality_scores["agreeableness"] > 75:
        return "HR or Leadership Roles"

    if personality_scores["conscientiousness"] > 80:
        return "Software Engineer or Architect"

    return "Technology Professional"

# INDUSTRY COMPATIBILITY ENGINE

def industry_comparison(personality_scores):

    industry_benchmarks = {
        "Technology": {"openness": 75, "conscientiousness": 80},
        "Business": {"extraversion": 75, "agreeableness": 70},
        "Research": {"openness": 85, "conscientiousness": 75},
        "Creative": {"openness": 80, "extraversion": 65},
    }

    comparison = {}

    for industry, traits in industry_benchmarks.items():
        score = 0
        for trait, benchmark in traits.items():
            score += 100 - abs(personality_scores[trait] - benchmark)
        comparison[industry] = score // len(traits)

    return comparison
