def get_recommendations(apnea_severity, snoring_severity):
    """
    Generate personalized recommendations based on apnea and snoring severity.
    
    Args:
        apnea_severity (str): Severity of sleep apnea ("Normal", "Mild", "Moderate", "Severe").
        snoring_severity (str): Severity of snoring ("Low", "Moderate", "High").
        
    Returns:
        str: Formatted recommendations as markdown bullet points.
    """
    recommendations = []

    if apnea_severity in ["Moderate", "Severe"]:
        recommendations.extend([
            "Consult a sleep specialist for professional evaluation",
            "Consider a sleep study for detailed diagnosis",
            "Discuss CPAP therapy options with your healthcare provider"
        ])

    if snoring_severity in ["Moderate", "High"]:
        recommendations.extend([
            "Consider positional therapy to reduce snoring",
            "Evaluate lifestyle factors (weight, alcohol, smoking)",
            "Discuss oral appliances with your dentist"
        ])

    if apnea_severity == "Mild" and snoring_severity == "Low":
        recommendations.extend([
            "Monitor sleep patterns regularly",
            "Practice good sleep hygiene",
            "Consider follow-up assessment in 6 months"
        ])

    return "\n".join(f"- {rec}" for rec in recommendations)