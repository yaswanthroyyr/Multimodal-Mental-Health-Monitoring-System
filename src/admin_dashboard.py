import streamlit as st
import pandas as pd
import altair as alt
from sqlalchemy import desc

# Import your database session and models
from database import SessionLocal, StudentProfile, ClassSession

st.set_page_config(page_title="Admin Console | AI Monitor", page_icon="🛡️", layout="wide")

st.title("🛡️ Central Admin Console (Multi-Modal)")
st.markdown("Monitor student mental health, visual engagement, and **vocal stress** in real-time.")

# --- 1. Fetch Registered Students ---
db = SessionLocal()
try:
    students = db.query(StudentProfile).all()
finally:
    db.close()

if not students:
    st.warning("⚠️ No students found in the database. Please run database.py to seed a test user.")
    st.stop()

# --- 2. Sidebar Navigation ---
st.sidebar.title("Classroom Roster")
student_dict = {f"{s.name} ({s.email})": s.student_id for s in students}
selected_label = st.sidebar.radio("Select Student to Monitor:", list(student_dict.keys()))
selected_student_id = student_dict[selected_label]

st.divider()
st.subheader(f"🔴 Live Telemetry Feed: {selected_label.split(' ')[0]}")

# --- 3. Real-Time Auto-Refreshing Fragment ---
@st.fragment(run_every="2s")
def render_live_dashboard():
    db = SessionLocal()
    try:
        # Fetch the last 60 seconds of data for the selected student
        sessions = db.query(ClassSession).filter(
            ClassSession.student_id == selected_student_id
        ).order_by(desc(ClassSession.date)).limit(60).all()
    finally:
        db.close()

    if not sessions:
        st.info("⏳ Waiting for student to connect to the portal and begin streaming...")
        return

    # Grab the absolute newest data point
    latest = sessions[0]
    
    # --- A. Top-Level Metric Cards (Now with 4 Columns!) ---
    col1, col2, col3, col4 = st.columns(4)
    
    # Color code the emotion
    emo_color = "normal"
    if latest.dominant_affect in ['Sad', 'Angry', 'Fear', 'Disgust']: emo_color = "inverse"
    
    # Color code engagement
    eng_color = "normal"
    if latest.engagement_score < 50: eng_color = "inverse"

    col1.metric("Psychological State", latest.dominant_affect, delta_color=emo_color)
    col2.metric("Visual Engagement", f"{latest.engagement_score}%", delta_color=eng_color)
    
    # Safely extract physical and vocal status from the JSONB column
    telemetry = latest.behavioral_telemetry or {}
    physical_status = telemetry.get("physical_status", "Unknown")
    vocal_status = telemetry.get("vocal_status", "Silent")
    speech_pace = telemetry.get("speech_pace", 0.0)

    col3.metric("Physical Behavior", physical_status)
    
    # NEW: Audio Metric Card
    pace_text = f"{speech_pace} pace" if speech_pace > 0 else ""
    col4.metric("Microphone Status", vocal_status, delta=pace_text, delta_color="off")

    # --- B. The Longitudinal Graph ---
    sessions.reverse()
    
    df = pd.DataFrame([{
        "Time": s.date.strftime("%H:%M:%S"),
        "Engagement": s.engagement_score,
        "Emotion": s.dominant_affect,
        "Vocal Status": s.behavioral_telemetry.get("vocal_status", "Silent") if s.behavioral_telemetry else "Silent"
    } for s in sessions])

    chart = alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('Time', title="Session Time"),
        y=alt.Y('Engagement', scale=alt.Scale(domain=[0, 100]), title="Engagement Score (%)"),
        color=alt.Color('Emotion', scale=alt.Scale(scheme='category10'), legend=alt.Legend(title="Detected Emotion")),
        
        # NEW: Added Vocal Status to the hover tooltip!
        tooltip=['Time', 'Emotion', 'Engagement', 'Vocal Status']
    ).properties(
        height=400
    ).interactive()

    # Note: width="stretch" fixes the deprecation warning from earlier!
    st.altair_chart(chart, width="stretch")

# Execute the fragment
render_live_dashboard()