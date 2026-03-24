import cv2
import numpy as np
import base64
import json
import os
import re
import random
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Import our custom AI and Database modules
from feature_extractor import MultimodalFeatureExtractor
from engagement_tracker import EngagementTracker
from audio_extractor import AudioStressExtractor
from database import SessionLocal, ClassSession, StudentProfile, init_db

# --- 1. Modern Server Lifespan & Model Loading ---
extractor = None
tracker = None
audio_tracker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global extractor, tracker, audio_tracker
    print("⏳ Booting Central AI Server... Loading Multimodal Models.")
    
    # Load AI Models into Memory
    extractor = MultimodalFeatureExtractor()
    tracker = EngagementTracker()
    audio_tracker = AudioStressExtractor()
    print("✅ SOTA Models, Vision Trackers, & Audio Engines Loaded!")
    
    # Safely verify database tables
    init_db()
    
    yield # Server is running
    print("🛑 Shutting down server and cleaning up models.")

app = FastAPI(title="AI Mental Health Server", lifespan=lifespan)

# --- 2. Data Models & OTP Memory ---
otp_store = {} # Temporarily stores {email: "123456"}
EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

class LoginRequest(BaseModel):
    email: str
    password: str

class OTPRequest(BaseModel):
    email: str

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    otp: str

# --- 3. Authentication & OTP Endpoints ---
@app.post("/request-otp")
async def request_otp(req: OTPRequest):
    """Generates an OTP and simulates sending an email."""
    if not re.match(EMAIL_REGEX, req.email):
        return JSONResponse(status_code=400, content={"detail": "Invalid email format."})
    
    db = SessionLocal()
    try:
        existing_user = db.query(StudentProfile).filter(StudentProfile.email == req.email).first()
        if existing_user:
            return JSONResponse(status_code=400, content={"detail": "Email is already registered."})
    finally:
        db.close()

    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))
    otp_store[req.email] = otp
    
    # Simulate sending the email by printing to the terminal
    print(f"\n" + "="*50)
    print(f"📧 [EMAIL SIMULATION] To: {req.email}")
    print(f"🔑 Subject: Centurion AI Monitor Verification")
    print(f"🔢 Your OTP Code is: {otp}")
    print("="*50 + "\n")
    
    return {"status": "success", "message": "OTP sent! Check your terminal."}

@app.post("/register")
async def register_user(req: RegisterRequest):
    """Verifies OTP and creates the new student profile."""
    if not re.match(EMAIL_REGEX, req.email):
        return JSONResponse(status_code=400, content={"detail": "Invalid email format."})
    
    # Verify the OTP
    stored_otp = otp_store.get(req.email)
    if not stored_otp or stored_otp != req.otp:
        return JSONResponse(status_code=400, content={"detail": "Invalid or expired OTP."})
    
    db = SessionLocal()
    try:
        new_student = StudentProfile(
            name=req.name,
            email=req.email,
            password=req.password
        )
        db.add(new_student)
        db.commit()
        db.refresh(new_student)
        
        # Clear the OTP now that it has been used
        del otp_store[req.email]
        
        print(f"🆕 Verified & Registered: {new_student.name} ({new_student.email})")
        return {"status": "success", "message": "Registration successful!"}
    finally:
        db.close()

@app.post("/login")
async def login_user(req: LoginRequest):
    """Checks credentials against PostgreSQL."""
    db = SessionLocal()
    try:
        user = db.query(StudentProfile).filter(StudentProfile.email == req.email).first()
        if user and user.password == req.password:
            print(f"🔐 Successful Login: {user.name} ({user.email})")
            return {"status": "success", "student_id": user.student_id, "name": user.name}
        else:
            print(f"❌ Failed Login Attempt: {req.email}")
            return JSONResponse(status_code=401, content={"detail": "Invalid email or password"})
    finally:
        db.close()

# --- 4. Serve the Student Webpage ---
@app.get("/")
async def serve_portal():
    """Serves the HTML portal directly from FastAPI."""
    with open("public/student_portal.html", "r") as f:  # <--- UPDATED PATH
        html_content = f.read()
    return HTMLResponse(content=html_content)

# --- 5. The Real-Time Multimodal WebSocket Endpoint ---
@app.websocket("/ws/stream/{student_id}")
async def websocket_endpoint(websocket: WebSocket, student_id: str):
    await websocket.accept()
    print(f"📡 Multimodal Stream Connected for Student: {student_id}")
    
    LABELS = ['Sad', 'Disgust', 'Angry', 'Neutral', 'Fear', 'Surprise', 'Happy']
    
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            audio_mfcc_data = []
            speech_pace = 0.0
            audio_status = "Silent"
            dominant_affect = "Neutral"
            telemetry = {
                "engagement_score": 0.0,
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "ear": 0.0,
                "status": "No Face"
            }

            # --- A. Process Audio Chunk (.WAV) ---
            if payload.get("audio_b64"):
                try:
                    raw_b64 = payload["audio_b64"]
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",", 1)[1]
                    
                    raw_b64 = re.sub(r'[^A-Za-z0-9+/]', '', raw_b64)
                    
                    if len(raw_b64) % 4 == 1:
                        raw_b64 = raw_b64[:-1]
                    
                    padding_needed = (4 - (len(raw_b64) % 4)) % 4
                    raw_b64 += "=" * padding_needed
                        
                    audio_bytes = base64.b64decode(raw_b64)
                    
                    if len(audio_bytes) > 500:
                        temp_audio_path = f"temp_{student_id}.wav"
                        
                        with open(temp_audio_path, "wb") as f:
                            f.write(audio_bytes)
                        
                        audio_results = audio_tracker.analyze_audio_chunk(temp_audio_path)
                        audio_mfcc_data = audio_results.get("audio_mfcc_profile", [])
                        speech_pace = audio_results.get("avg_speech_pace", 0.0)
                        audio_status = audio_results.get("status", "Silent")
                        
                        if os.path.exists(temp_audio_path):
                            os.remove(temp_audio_path)
                            
                except Exception as e:
                    print(f"⚠️ Audio decoding skipped: {e}")

            # --- B. Process Video Frame ---
            if payload.get("image_b64"):
                try:
                    img_data = base64.b64decode(payload["image_b64"].split(',')[1])
                    np_arr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        telemetry = tracker.analyze_frame(frame)
                        temp_img_path = f"temp_{student_id}.jpg"
                        cv2.imwrite(temp_img_path, frame)
                        
                        vid_probs = extractor.process_video_frame(temp_img_path)[0]
                        idx = torch.argmax(vid_probs).item()
                        dominant_affect = LABELS[idx]
                        
                        if os.path.exists(temp_img_path):
                            os.remove(temp_img_path)
                except Exception as e:
                    print(f"⚠️ Video processing skipped: {e}")
                
            # --- C. Fuse Data & Save to PostgreSQL ---
            db = SessionLocal()
            try:
                session_entry = ClassSession(
                    student_id=student_id,
                    dominant_affect=dominant_affect,
                    engagement_score=telemetry.get("engagement_score", 0.0),
                    behavioral_telemetry={
                        "head_yaw": telemetry.get("head_yaw", 0.0),
                        "head_pitch": telemetry.get("head_pitch", 0.0),
                        "ear": telemetry.get("ear", 0.0),
                        "physical_status": telemetry.get("status", "Unknown"),
                        "vocal_status": audio_status,
                        "speech_pace": speech_pace
                    },
                    audio_mfcc_profile=audio_mfcc_data if audio_mfcc_data else None
                )
                db.add(session_entry)
                db.commit()
            finally:
                db.close()
                
            # --- D. Send Real-Time Feedback to Student UI ---
            await websocket.send_text(json.dumps({
                "status": "Processed",
                "engagement": telemetry.get("engagement_score", 0.0),
                "emotion": dominant_affect,
                "audio_status": audio_status
            }))
                
    except WebSocketDisconnect:
        print(f"🛑 Stream Disconnected for Student: {student_id}")
    except Exception as e:
        print(f"❌ Critical Processing Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)