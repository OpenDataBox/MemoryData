CONVERSATION_PROFILE_PART3_EXTRACTION_PROMPT = """
Please analyze the latest user-AI conversation below and update the user profile based on the 90 personality preference dimensions.

Here are the 90 dimensions and their explanations:

[Psychological Model (Basic Needs & Personality)]
Extraversion: Preference for social activities.
Openness: Willingness to embrace new ideas and experiences.
Agreeableness: Tendency to be friendly and cooperative.
Conscientiousness: Responsibility and organizational ability.
Neuroticism: Emotional stability and sensitivity.
Physiological Needs: Concern for comfort and basic needs.
Need for Security: Emphasis on safety and stability.
Need for Belonging: Desire for group affiliation.
Need for Self-Esteem: Need for respect and recognition.
Cognitive Needs: Desire for knowledge and understanding.
Aesthetic Appreciation: Appreciation for beauty and art.
Self-Actualization: Pursuit of one's full potential.
Need for Order: Preference for cleanliness and organization.
Need for Autonomy: Preference for independent decision-making and action.
Need for Power: Desire to influence or control others.
Need for Achievement: Value placed on accomplishments.

[AI Alignment Dimensions]
Helpfulness: Whether the AI's response is practically useful to the user. (This reflects user's expectation of AI)
Honesty: Whether the AI's response is truthful. (This reflects user's expectation of AI)
Safety: Avoidance of sensitive or harmful content. (This reflects user's expectation of AI)
Instruction Compliance: Strict adherence to user instructions. (This reflects user's expectation of AI)
Truthfulness: Accuracy and authenticity of content. (This reflects user's expectation of AI)
Coherence: Clarity and logical consistency of expression. (This reflects user's expectation of AI)
Complexity: Preference for detailed and complex information.
Conciseness: Preference for brief and clear responses.

[Content Platform Interest Tags]
Science Interest: Interest in science topics.
Education Interest: Concern with education and learning.
Psychology Interest: Interest in psychology topics.
Family Concern: Interest in family and parenting.
Fashion Interest: Interest in fashion topics.
Art Interest: Engagement with or interest in art.
Health Concern: Concern with physical health and lifestyle.
Financial Management Interest: Interest in finance and budgeting.
Sports Interest: Interest in sports and physical activity.
Food Interest: Passion for cooking and cuisine.
Travel Interest: Interest in traveling and exploring new places.
Music Interest: Interest in music appreciation or creation.
Literature Interest: Interest in literature and reading.
Film Interest: Interest in movies and cinema.
Social Media Activity: Frequency and engagement with social media.
Tech Interest: Interest in technology and innovation.
Environmental Concern: Attention to environmental and sustainability issues.
History Interest: Interest in historical knowledge and topics.
Political Concern: Interest in political and social issues.
Religious Interest: Interest in religion and spirituality.
Gaming Interest: Enjoyment of video games or board games.
Animal Concern: Concern for animals or pets.
Emotional Expression: Preference for direct vs. restrained emotional expression.
Sense of Humor: Preference for humorous or serious communication style.
Information Density: Preference for detailed vs. concise information.
Language Style: Preference for formal vs. casual tone.
Practicality: Preference for practical advice vs. theoretical discussion.

**Task Instructions:**
1. Review the existing user profile below
2. Analyze the new conversation for evidence of the 90 dimensions above
3. Update and integrate the findings into a comprehensive user profile
4. For each dimension that can be identified, use the format: Dimension ( Level(High/Medium/Low) )
5. Include brief reasoning for each dimension when possible
6. Maintain existing insights from the old profile while incorporating new observations
7. If a dimension cannot be inferred from either the old profile or new conversation, do not include it
 
Output Requirements:
- Return ONLY one fenced JSON code block (```json ... ```), no extra text outside the code block.
- Use ASCII quotes only (no smart quotes).
- Evidence format: prefer "[conversation_id:EVENT_ID]" or raw "EVENT_ID" that appears in the conversation memcells.
- Include only observed dimensions; omit unknown or unsupported ones.
- The fields must align with the existing schema used by the system.

JSON Template:
```json
{
  "user_profiles": [
    {
      "user_id": "USER_ID",
      "user_name": "USER_NAME",
      "personality": [],
      "way_of_decision_making": [],
      "working_habit_preference": [],
      "interests": [],
      "tendency": [],
      "motivation_system": [],
      "fear_system": [],
      "value_system": [],
      "humor_use": [],
      "colloquialism": [],
      "output_reasoning": ""
    }
  ]
}
```

Field Item Schema (for list fields):
- Each item must be: {"value": string, "evidences": [string], "level": string?}
"""
