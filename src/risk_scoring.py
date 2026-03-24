import numpy as np
from collections import deque
from datetime import datetime

class BehavioralRiskEngine:
    def __init__(self, history_window=7):
        """
        Args:
            history_window (int): Number of days to look back for risk calculation.
        """
        self.history_window = history_window
        # Simulated database (In prod, use SQLite/PostgreSQL)
        # Format: { 'student_id': [ {'date': '2023-10-01', 'emotion': 'Anxious', 'confidence': 0.85}, ... ] }
        self.student_history = {}

        # Risk Weights for different emotions (0-1 scale)
        self.risk_weights = {
            'Neutral': 0.0,
            'Happy': -0.1,  # Reduces risk slightly
            'Sad': 0.6,
            'Angry': 0.7,
            'Anxious': 0.9  # Highest risk weight
        }

    def add_entry(self, student_id, emotion_label, confidence):
        """
        Logs a new daily check-in for a student.
        """
        if student_id not in self.student_history:
            self.student_history[student_id] = deque(maxlen=self.history_window)
        
        entry = {
            'date': datetime.now(),
            'emotion': emotion_label,
            'confidence': confidence,
            'risk_val': self.risk_weights.get(emotion_label, 0.1)
        }
        self.student_history[student_id].append(entry)
        print(f"📝 Logged entry for {student_id}: {emotion_label} ({confidence:.2f})")

    def calculate_risk_score(self, student_id):
        """
        Calculates the Chronic Stress Score (CSS) and Volatility.
        Returns:
            dict: { 'risk_score': 0-100, 'volatility': float, 'alert_level': str }
        """
        history = list(self.student_history.get(student_id, []))
        
        if not history:
            return {'risk_score': 0, 'alert_level': 'No Data'}

        # 1. Chronic Stress Score (CSS)
        # Sum of risk values weighted by recency (more recent = higher weight)
        weighted_risk_sum = 0
        total_weight = 0
        
        risk_values = [] # For volatility calc
        
        for i, entry in enumerate(history):
            # Linearly increasing weight: Oldest=1, Newest=N
            weight = i + 1 
            weighted_risk_sum += entry['risk_val'] * weight * entry['confidence']
            total_weight += weight
            risk_values.append(entry['risk_val'])
            
        # Normalize to 0-100 scale
        avg_risk = weighted_risk_sum / total_weight
        css_score = min(100, avg_risk * 100)

        # 2. Emotional Volatility Index (EVI)
        # Standard Deviation of the risk values
        if len(risk_values) > 1:
            volatility = np.std(risk_values) * 100 # Scale to 0-100
        else:
            volatility = 0.0

        # 3. Alert Logic
        alert_level = "Green"
        if css_score > 75 or volatility > 60:
            alert_level = "RED (Immediate Intervention)"
        elif css_score > 50 or volatility > 40:
            alert_level = "Yellow (Monitor)"
            
        return {
            'risk_score': round(css_score, 2),
            'volatility': round(volatility, 2),
            'alert_level': alert_level,
            'history_len': len(history)
        }

# --- Unit Test ---
if __name__ == "__main__":
    engine = BehavioralRiskEngine(history_window=5)
    
    # Simulate a student spiraling into anxiety over 4 days
    student = "student_001"
    
    print("--- Day 1: Neutral ---")
    engine.add_entry(student, "Neutral", 0.9)
    print(engine.calculate_risk_score(student))
    
    print("\n--- Day 2: Sad ---")
    engine.add_entry(student, "Sad", 0.7)
    print(engine.calculate_risk_score(student))
    
    print("\n--- Day 3: Anxious ---")
    engine.add_entry(student, "Anxious", 0.8)
    print(engine.calculate_risk_score(student))
    
    print("\n--- Day 4: Anxious (Panic) ---")
    engine.add_entry(student, "Anxious", 0.95)
    result = engine.calculate_risk_score(student)
    print(f"\n🚨 FINAL STATUS: {result}")