from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session as flask_session
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os
import secrets
import logging
from wtforms import Form, StringField, PasswordField, BooleanField, SubmitField, TextAreaField, SelectField, DecimalField, IntegerField, DateTimeField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError, Optional
from flask_wtf import FlaskForm
import socket
import requests
from OpenSSL import crypto
import io
import numpy as np



# Firebase imports
from firebase_config import (
    User, Session, create_user, get_user_by_id, get_user_by_email, update_user,
    create_session as fb_create_session, get_session_by_id, update_session, delete_session,
    get_sessions_by_trainer, get_sessions_by_learner, get_available_sessions,
    count_sessions_by_trainer, create_booking, get_bookings_by_status,
    create_review, get_reviews_by_trainer, count_users, db
)

# ML dependencies
try:
    import librosa
    import soundfile as sf
except Exception:
    librosa = None
    sf = None

try:
    from pydub import AudioSegment
    import pydub.utils
    pydub.utils.which = lambda x: r'C:\Users\rakes\anaconda\envs\mlenv\Library\bin\ffmpeg.exe' if 'ffmpeg' in x else r'C:\Users\rakes\anaconda\envs\mlenv\Library\bin\ffprobe.exe'
    AudioSegment.converter = r'C:\Users\rakes\anaconda\envs\mlenv\Library\bin\ffmpeg.exe'
    AudioSegment.ffprobe = r'C:\Users\rakes\anaconda\envs\mlenv\Library\bin\ffprobe.exe'
except Exception:
    AudioSegment = None

try:
    from tensorflow import keras
except Exception:
    keras = None

# CREATE FLASK APP
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here-change-in-production')
app.config['PREFERRED_URL_SCHEME'] = 'https'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
app.logger.setLevel(logging.DEBUG)

# LOAD ML MODELS
emotion_model = None
face_emotion_model = None

if keras is not None:                
    emotion_model = keras.Sequential([
        keras.layers.Input(shape=(40, 1)),
        keras.layers.Conv1D(128, 5, padding='same'),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.1),
        keras.layers.MaxPooling1D(pool_size=8),
        keras.layers.Conv1D(128, 5, padding='same'),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Flatten(),
        keras.layers.Dense(8),
        keras.layers.Activation('softmax')
    ])

    # Load only the weights (not the full model)
    emotion_model.load_weights('Emotion_Voice_Detection_Model.h5')

    # Compile the model
    emotion_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    app.logger.info("✅ Voice Model weights loaded")
    print("[SUCCESS] for voice model")

    # Log model info
    app.logger.info(f"Model input shape: {emotion_model.input_shape}")
    app.logger.info(f"Model output shape: {emotion_model.output_shape}")


# FER-based face emotion detector (replaces YOLO model)
import threading
_fer_lock = threading.Lock()  # FER/Keras is NOT thread-safe; serialize all inference calls
try:
    from fer.fer import FER  # fer.__init__ does not re-export FER; class lives in fer.fer submodule
    # mtcnn=False uses OpenCV Haar cascade — much faster for real-time frames and thread-friendlier
    face_emotion_model = FER(mtcnn=False)
    app.logger.info("✅ FER face emotion detector loaded (OpenCV cascade)")
    print("[SUCCESS] FER face emotion model loaded successfully")
    # WARM-UP: run one inference in the main thread to initialize the Keras/TF graph.
    # Without this, the first call from a worker thread can fail silently.
    _warmup_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _warmup_result = face_emotion_model.detect_emotions(_warmup_img)
    print(f"[FER WARMUP] Inference OK, result: {_warmup_result}")
    app.logger.info(f"FER warmup inference completed: {_warmup_result}")
    del _warmup_img, _warmup_result
except ImportError as e:
    app.logger.warning(f'FER not available: {e}')
    print(f"[WARNING] FER import failed: {e}")
    face_emotion_model = None
except Exception as e:
    app.logger.exception('Failed to load FER face emotion model: %s', e)
    print(f"[ERROR] FER model error: {e}")
    face_emotion_model = None

# At the top of your imports, ensure you have:
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False
    app.logger.warning('ReportLab not available for PDF generation')

# Initialize extensions
socketio = SocketIO(app, 
                    cors_allowed_origins="*", 
                    async_mode='threading',
                    logger=True,
                    engineio_logger=True,
                    ping_timeout=60,
                    ping_interval=25)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'


def _process_audio_bytes(audio_bytes):
    """Return mfcc feature vector (1,40,1) from WAV audio."""
    try:
        import wave
        import struct
        
        wav_io = io.BytesIO(audio_bytes)
        with wave.open(wav_io, 'rb') as wav:
            sr = wav.getframerate()
            n_frames = wav.getnframes()
            audio_data = wav.readframes(n_frames)
            
            if wav.getsampwidth() == 2:
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif wav.getsampwidth() == 4:
                audio_np = np.frombuffer(audio_data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio_np = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            
            if wav.getnchannels() == 2:
                audio_np = audio_np[::2]
            
            if librosa is None:
                raise Exception('librosa is required for audio processing')
            
            mfccs = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs, axis=1)
            features = mfccs_mean.reshape(1, 40, 1).astype('float32')
            
            app.logger.info('✓ Processed WAV audio: sr=%d, frames=%d, duration=%.2fs', sr, n_frames, n_frames/sr)
            return features, sr
            
    except Exception as e:
        app.logger.exception('WAV processing failed: %s', e)
        raise Exception(f'Real audio processing failed: {e}')


@login_manager.user_loader
def load_user(user_id):
    user_data = get_user_by_id(user_id)
    if user_data:
        return User(user_data)
    return None

# Forms
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    first_name = StringField('First Name', validators=[DataRequired(), Length(max=50)])
    last_name = StringField('Last Name', validators=[DataRequired(), Length(max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    user_type = SelectField('I want to join as a:', choices=[
        ('learner', 'Learner'),
        ('trainer', 'Trainer')
    ], validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_email(self, email):
        user = get_user_by_email(email.data)
        if user:
            raise ValidationError('Email already registered. Please use a different email.')

class SessionForm(FlaskForm):
    title = StringField('Session Title', validators=[DataRequired(), Length(max=200)])
    description = TextAreaField('Description', validators=[DataRequired()])
    category = SelectField('Category', choices=[
        ('programming', 'Programming'),
        ('design', 'Design'),
        ('business', 'Business'),
        ('marketing', 'Marketing'),
        ('language', 'Language'),
        ('music', 'Music'),
        ('fitness', 'Fitness'),
        ('academic', 'Academic'),
        ('other', 'Other')
    ], validators=[DataRequired()])
    scheduled_time = DateTimeField('Date & Time', format='%Y-%m-%dT%H:%M', validators=[DataRequired()])
    duration = IntegerField('Duration (minutes)', validators=[DataRequired()])
    price = DecimalField('Price ($)', places=2, validators=[Optional()])
    max_participants = IntegerField('Max Participants', default=1, validators=[DataRequired()])
    difficulty = SelectField('Difficulty Level', choices=[
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced')
    ], validators=[DataRequired()])
    prerequisites = TextAreaField('Prerequisites', validators=[Optional()])
    materials = TextAreaField('Required Materials', validators=[Optional()])
    is_recurring = BooleanField('This is a recurring session')
    recurrence_pattern = SelectField('Recurrence Pattern', choices=[
        ('weekly', 'Weekly'),
        ('biweekly', 'Bi-weekly'),
        ('monthly', 'Monthly')
    ], validators=[Optional()])
    recurrence_count = IntegerField('Number of Sessions', default=2, validators=[Optional()])
    submit = SubmitField('Create Session')

class ProfileForm(FlaskForm):
    first_name = StringField('First Name', validators=[DataRequired(), Length(max=50)])
    last_name = StringField('Last Name', validators=[DataRequired(), Length(max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone = StringField('Phone Number', validators=[Optional(), Length(max=20)])
    bio = TextAreaField('Bio', validators=[Optional(), Length(max=500)])
    specialization = StringField('Specialization', validators=[Optional(), Length(max=100)])
    hourly_rate = DecimalField('Hourly Rate ($)', places=2, validators=[Optional()])
    experience = IntegerField('Years of Experience', validators=[Optional()])
    timezone = SelectField('Timezone', choices=[
        ('UTC', 'UTC'),
        ('EST', 'Eastern Time'),
        ('PST', 'Pacific Time'),
        ('CST', 'Central Time'),
        ('GMT', 'GMT'),
        ('CET', 'Central European Time')
    ], validators=[DataRequired()])
    submit = SubmitField('Update Profile')

class ChangePasswordForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired(), Length(min=6)])
    confirm_new_password = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField('Change Password')

# Jitsi Helper Functions
def generate_room_name():
    return f"DirectProf-{secrets.token_urlsafe(12)}"

def get_jitsi_server_url():
    servers = [
        "https://meet.jit.si",
        "https://8x8.vc",
        "https://meet.opensight.chat"
    ]
    return servers[0]

def check_jitsi_connectivity():
    server = get_jitsi_server_url()
    try:
        response = requests.get(server, timeout=5)
        return response.status_code == 200
    except:
        return False

def generate_session_report(session_id):
    """Generate PDF report for completed session"""
    if not PDF_AVAILABLE:
        app.logger.warning('PDF generation not available - reportlab not installed')
        return None
    
    try:
        # Get session data
        sess_data = get_session_by_id(session_id)
        if not sess_data:
            app.logger.error(f'Session {session_id} not found for report generation')
            return None
        
        session = Session(sess_data)
        trainer = get_user_by_id(session.trainer_id)
        if not trainer:
            app.logger.error(f'Trainer {session.trainer_id} not found for report')
            return None
        
        learner = None
        if session.learner_id:
            learner_data = get_user_by_id(session.learner_id)
            if learner_data:
                learner = learner_data
        
        # Get detection records - ensure session_id is a string to match how it's stored
        try:
            session_id_str = str(session_id)
            app.logger.info(f'Fetching detection records for report generation, session_id: {session_id_str}')
            
            detection_records = db.collection('detection_records')\
                .where('session_id', '==', session_id_str)\
                .stream()
            
            records = []
            for doc in detection_records:
                record = doc.to_dict()
                records.append(record)
            
            # Sort by timestamp
            records.sort(key=lambda x: x.get('timestamp', datetime.min) if x.get('timestamp') else datetime.min)
            app.logger.info(f'Found {len(records)} detection records for session {session_id_str}')
        except Exception as e:
            app.logger.error(f'Error fetching detection records for report: {e}')
            import traceback
            app.logger.error(traceback.format_exc())
            records = []
        
        # Create PDF directory
        os.makedirs('session_reports', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f'session_reports/session_{session_id}_{timestamp}.pdf'
        
        # Create the PDF
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        
        # Container for PDF elements
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        elements.append(Paragraph('Session Detection Report', title_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Session Info
        session_info = [
            ['Session Title:', session.title if hasattr(session, 'title') else 'N/A'],
            ['Trainer:', f"{trainer.get('first_name', '')} {trainer.get('last_name', '')}"],
            ['Learner:', f"{learner.get('first_name', '')} {learner.get('last_name', '')}" if learner else 'N/A'],
            ['Date:', session.scheduled_time.strftime('%Y-%m-%d %H:%M') if hasattr(session, 'scheduled_time') and session.scheduled_time else 'N/A'],
            ['Duration:', f"{session.duration} minutes" if hasattr(session, 'duration') else 'N/A'],
            ['Category:', session.category if hasattr(session, 'category') else 'N/A'],
            ['Status:', session.status if hasattr(session, 'status') else 'N/A']
        ]
        
        session_table = Table(session_info, colWidths=[2*inch, 4*inch])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(session_table)
        elements.append(Spacer(1, 0.5*inch))
        
        # Detection Summary
        elements.append(Paragraph('Detection Summary', styles['Heading2']))
        elements.append(Spacer(1, 0.2*inch))
        
        if records:
            # Calculate statistics
            audio_records = [r for r in records if r.get('detection_type') == 'audio']
            video_records = [r for r in records if r.get('detection_type') == 'video']
            
            # Emotion distribution
            emotions = {}
            for r in audio_records:
                emotion_data = r.get('audio_emotion', {})
                if isinstance(emotion_data, dict):
                    emotion = emotion_data.get('primary', 'unknown')
                    emotions[emotion] = emotions.get(emotion, 0) + 1
            
            # Average stress
            stress_scores = []
            for r in audio_records:
                stress_data = r.get('voice_stress', {})
                if isinstance(stress_data, dict):
                    score = stress_data.get('score', 0)
                    if score > 0:
                        stress_scores.append(score)
            
            avg_stress = sum(stress_scores) / len(stress_scores) if stress_scores else 0
            
            # Gestures detected
            all_gestures = []
            for r in video_records:
                gestures = r.get('gestures', [])
                if isinstance(gestures, list):
                    all_gestures.extend(gestures)
            
            summary_data = [
                ['Metric', 'Value'],
                ['Total Detection Records', str(len(records))],
                ['Audio Detections', str(len(audio_records))],
                ['Video Detections', str(len(video_records))],
                ['Average Voice Stress', f"{avg_stress*100:.1f}%" if stress_scores else 'N/A'],
                ['Most Common Emotion', max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'N/A'],
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 0.3*inch))
            
            # Detailed Records (limited to first 20 for readability)
            elements.append(Paragraph('Detailed Detection Timeline (First 20 Records)', styles['Heading2']))
            elements.append(Spacer(1, 0.2*inch))
            
            detail_data = [['Time', 'Type', 'Emotion', 'Stress', 'Gestures']]
            for r in records[:20]:
                timestamp = r.get('timestamp')
                time_str = ''
                if timestamp:
                    if isinstance(timestamp, datetime):
                        time_str = timestamp.strftime('%H:%M:%S')
                    elif isinstance(timestamp, str):
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_str = dt.strftime('%H:%M:%S')
                        except:
                            time_str = 'N/A'
                else:
                    time_str = 'N/A'
                
                det_type = r.get('detection_type', 'N/A')
                
                if det_type == 'audio':
                    emotion_data = r.get('audio_emotion', {})
                    emotion = emotion_data.get('primary', 'N/A') if isinstance(emotion_data, dict) else 'N/A'
                    stress_data = r.get('voice_stress', {})
                    stress = f"{stress_data.get('score', 0)*100:.0f}%" if isinstance(stress_data, dict) else '-'
                    gestures = '-'
                else:
                    emotion_data = r.get('face_emotion', {})
                    emotion = emotion_data.get('emotion', 'N/A') if isinstance(emotion_data, dict) else 'N/A'
                    stress = '-'
                    gestures_list = r.get('gestures', [])
                    gestures = ', '.join(gestures_list) if isinstance(gestures_list, list) and gestures_list else 'None'
                
                detail_data.append([time_str, det_type, emotion, stress, gestures])
            
            if len(detail_data) > 1:  # If we have data beyond the header
                detail_table = Table(detail_data, colWidths=[1*inch, 0.8*inch, 1.2*inch, 0.8*inch, 1.5*inch])
                detail_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
                ]))
                elements.append(detail_table)
            else:
                elements.append(Paragraph('No detailed records available.', styles['Normal']))
        else:
            elements.append(Paragraph('No detection data recorded for this session.', styles['Normal']))
        
        # Add footer with generation timestamp
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(f'Report generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                                ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER)))
        
        # Build PDF
        doc.build(elements)
        app.logger.info(f'✓ PDF report generated: {pdf_filename}')
        
        return pdf_filename
        
    except Exception as e:
        app.logger.exception(f'Error generating PDF report: {e}')
        return None

def _append_detection_log(session_id, source, message, details=None):
    """Helper function to append detection logs"""
    try:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'source': source,
            'message': message,
            'details': details or {}
        }
        app.logger.info(f'Detection log: {message}')
        return log_entry
    except Exception as e:
        app.logger.error(f'Failed to create detection log: {e}')
        return {}

# Detection cache
DETECTION_CACHE = {}
LAST_FRAME_CACHE = {}
FACE_EMOTION_HISTORY = {}

# Gesture detection
try:
    import cv2
    import threading
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    _mp_lock = threading.Lock()
    # Cache the model globally. Loading this graph takes ~500ms, doing it per frame causes massive latency
    global_holistic = mp_holistic.Holistic(static_image_mode=True, model_complexity=0, min_detection_confidence=0.1)
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False
    app.logger.warning('MediaPipe not available for gesture detection')

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False
    app.logger.warning('ReportLab not available for PDF generation')

def calculate_eye_aspect_ratio(eye_landmarks):
    v1 = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - 
                        np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    v2 = np.linalg.norm(np.array([eye_landmarks[2].x, eye_landmarks[2].y]) - 
                        np.array([eye_landmarks[4].x, eye_landmarks[4].y]))
    h = np.linalg.norm(np.array([eye_landmarks[0].x, eye_landmarks[0].y]) - 
                       np.array([eye_landmarks[3].x, eye_landmarks[3].y]))
    return (v1 + v2) / (2.0 * h)

def detect_face_emotion(img, session_id=None):
    """Detect face emotion using the FER library (OpenCV cascade + emotion CNN)."""
    try:
        if face_emotion_model is None:
            return 'not_detected', 0.0

        if img is None or img.size == 0:
            app.logger.warning('detect_face_emotion: received empty/None image')
            return 'not_detected', 0.0

        h, w = img.shape[:2]
        app.logger.debug(f'detect_face_emotion: img shape={img.shape}, dtype={img.dtype}')

        # FER expects BGR image (OpenCV default) — no conversion needed
        # Slight brightness boost helps in dim lighting
        img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)

        # FER/Keras is NOT thread-safe. Use lock to serialize inference across concurrent frame uploads.
        with _fer_lock:
            detections = face_emotion_model.detect_emotions(img_enhanced)

        app.logger.debug(f'detect_face_emotion: raw detections={detections}')

        if not detections:
            # Log WHY: run bare Haar cascade to see if face was found at all
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            import cv2 as _cv2
            fc = _cv2.CascadeClassifier(_cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            haar_faces = fc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            app.logger.debug(f'detect_face_emotion: Haar saw {len(haar_faces)} faces, FER returned empty')
            print(f"[INFO] FER: no face ({len(haar_faces)} Haar faces) - Session: {session_id}")
            return 'not_detected', 0.0

        # Pick the detection with the largest face bounding box (closest to camera)
        best = max(detections, key=lambda d: d['box'][2] * d['box'][3])
        emotions_dict = best['emotions']  # e.g. {'angry': 0.02, 'happy': 0.91, ...}

        # Get the dominant emotion and its confidence
        emotion = max(emotions_dict, key=emotions_dict.get)
        confidence = float(emotions_dict[emotion])

        # Smoothing: keep a rolling window of the last 5 detections per session
        if session_id and confidence > 0.15:
            if session_id not in FACE_EMOTION_HISTORY:
                FACE_EMOTION_HISTORY[session_id] = []

            FACE_EMOTION_HISTORY[session_id].append((emotion, confidence))
            FACE_EMOTION_HISTORY[session_id] = FACE_EMOTION_HISTORY[session_id][-5:]

            if len(FACE_EMOTION_HISTORY[session_id]) >= 2:
                recent_emotions = [e for e, c in FACE_EMOTION_HISTORY[session_id]]
                emotion = max(set(recent_emotions), key=recent_emotions.count)
                confidence = sum(
                    c for e, c in FACE_EMOTION_HISTORY[session_id] if e == emotion
                ) / len([e for e in recent_emotions if e == emotion])

        app.logger.info(f'✓ Face: {emotion} ({confidence:.2f})')
        print(f"[FACE] Emotion: {emotion} (confidence: {confidence:.2f}) - Session: {session_id}")
        return emotion, confidence

    except Exception as e:
        app.logger.exception(f'Face emotion detection failed: {e}')
        print(f"[ERROR] Face detection error for session {session_id}: {e}")
        return 'not_detected', 0.0

def detect_gestures_from_frame(frame_data):
    if not MEDIAPIPE_AVAILABLE:
        return [], None, 0.0
    
    try:
        import base64
        img_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if not hasattr(detect_gestures_from_frame, 'saved_debug_frame'):
            cv2.imwrite('debug_frame.jpg', img)
            detect_gestures_from_frame.saved_debug_frame = True
            app.logger.info(f'DEBUG: Saved frame to debug_frame.jpg - shape: {img.shape if img is not None else "None"}')
        
        if img is None or img.size == 0:
            app.logger.warning('Decoded image is empty or None')
            return [], None, 0.0
        
        session_id = getattr(detect_gestures_from_frame, 'current_session_id', None)
        face_emotion, face_confidence = detect_face_emotion(img, session_id)
        app.logger.debug(f'Frame processing: face_emotion={face_emotion}, confidence={face_confidence:.2f}')
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        behaviors = []
        
        # Lock around the global MediaPipe instance since it is not thread-safe (same as FER)
        with _mp_lock:
            res = global_holistic.process(img_rgb)
            
        app.logger.debug('MediaPipe detection: face=%s, pose=%s, left_hand=%s, right_hand=%s, img_shape=%s',
                       bool(res.face_landmarks), bool(res.pose_landmarks),
                       bool(res.left_hand_landmarks), bool(res.right_hand_landmarks),
                       img_rgb.shape if img_rgb is not None else 'None')
            
        if res.face_landmarks:
            face = res.face_landmarks.landmark
            left_eye = [face[33], face[160], face[158], face[133], face[153], face[144]]
            right_eye = [face[362], face[385], face[387], face[263], face[373], face[380]]
            
            left_ear = calculate_eye_aspect_ratio(left_eye)
            right_ear = calculate_eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            if avg_ear < 0.25:
                behaviors.append('Blink')
        
        if res.left_hand_landmarks or res.right_hand_landmarks:
            hands = []
            if res.left_hand_landmarks:
                hands.append(res.left_hand_landmarks.landmark)
            if res.right_hand_landmarks:
                hands.append(res.right_hand_landmarks.landmark)
            
            for hand in hands:
                thumb_tip = hand[4]
                thumb_mcp = hand[2]
                index_tip = hand[8]
                
                if thumb_tip.y < thumb_mcp.y - 0.1 and index_tip.y > hand[6].y:
                    behaviors.append('Thumbs Up')
            
            if res.face_landmarks:
                face = res.face_landmarks.landmark
                mouth_center = np.array([(face[13].x + face[14].x)/2, (face[13].y + face[14].y)/2, face[13].z])
                chin = np.array([face[152].x, face[152].y, face[152].z])
                
                for hand in hands:
                    fingertips = [hand[8], hand[12], hand[16], hand[20]]
                    for tip in fingertips:
                        tip_np = np.array([tip.x, tip.y, tip.z])
                        if np.linalg.norm(tip_np - mouth_center) < 0.08:
                            behaviors.append('Biting Nails')
                            break
                    
                    mcp = np.array([hand[5].x, hand[5].y, hand[5].z])
                    wrist = np.array([hand[0].x, hand[0].y, hand[0].z])
                    if np.linalg.norm(mcp - chin) < 0.15 or np.linalg.norm(wrist - chin) < 0.15:
                        behaviors.append('Hand on Jaw')

        # Hand Raising: wrist Y must be clearly above nose Y
        # (In MediaPipe, Y=0=top of screen, so wrist.y < nose.y means wrist is higher)
        # Threshold 0.05 works well for typical seated webcam angles
        if (res.left_hand_landmarks or res.right_hand_landmarks) and res.face_landmarks:
            nose = res.face_landmarks.landmark[1]
            raise_hands = []
            if res.left_hand_landmarks:
                raise_hands.append(res.left_hand_landmarks.landmark)
            if res.right_hand_landmarks:
                raise_hands.append(res.right_hand_landmarks.landmark)
            for hand in raise_hands:
                wrist = hand[0]
                diff = nose.y - wrist.y  # positive = wrist is above nose
                app.logger.debug(f'Hand raise check: nose.y={nose.y:.3f} wrist.y={wrist.y:.3f} diff={diff:.3f}')
                print(f'[GESTURE DEBUG] Hand raise: nose.y={nose.y:.3f} wrist.y={wrist.y:.3f} diff={diff:.3f}')
                if diff > 0.05:  # wrist is at least 5% of frame height above the nose
                    behaviors.append('Raising Hand')
                    break

        # Leaning Forward: if shoulders are unusually high (close to nose Y) it signals forward lean
        if res.pose_landmarks and res.face_landmarks:
            nose = res.face_landmarks.landmark[1]
            left_shoulder = res.pose_landmarks.landmark[11]
            right_shoulder = res.pose_landmarks.landmark[12]
            avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
            # Shoulders closer to top of frame than normal = user leaning in (engaged)
            if avg_shoulder_y < nose.y + 0.05:
                behaviors.append('Leaning Forward')
        
        return list(set(behaviors)), face_emotion, face_confidence
    except Exception as e:
        app.logger.warning('Gesture detection failed: %s', e)
        return [], None, 0.0

# API Routes
@app.route('/api/session/<session_id>/detection', methods=['GET'])
@login_required
def get_session_detection(session_id):
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        return jsonify({'success': False, 'message': 'Session not found'}), 404
    
    sess = Session(sess_data)
    app.logger.info("detection request by user_id=%s for session_id=%s", current_user.get_id(), session_id)

    if current_user.id != sess.trainer_id:
        app.logger.warning("detection GET denied to non-trainer user_id=%s session_id=%s", current_user.get_id(), session_id)
        return jsonify({'success': False, 'message': 'Access denied'}), 403

    if session_id in DETECTION_CACHE:
        return jsonify({'success': True, 'detection': DETECTION_CACHE[session_id]})

    return jsonify({'success': False, 'message': 'No detection data available'}), 404

@app.route('/api/session/<session_id>/detection/live', methods=['GET'])
@login_required
def get_live_detection(session_id):
    """Real-time detection endpoint for trainers"""
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        return jsonify({'success': False, 'message': 'Session not found'}), 404
    
    sess = Session(sess_data)
    if current_user.id != sess.trainer_id:
        return jsonify({'success': False, 'message': 'Access denied - trainers only'}), 403
    
    # Get latest detection from cache
    detection_data = DETECTION_CACHE.get(session_id, {})
    
    # Also get recent detection records from database - ensure session_id is a string
    try:
        session_id_str = str(session_id)
        recent_records = db.collection('detection_records')\
            .where('session_id', '==', session_id_str)\
            .order_by('timestamp', direction='DESCENDING')\
            .limit(5)\
            .stream()
        
        records = [doc.to_dict() for doc in recent_records]
        
        return jsonify({
            'success': True,
            'live_detection': detection_data,
            'recent_records': records,
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        app.logger.error(f'Error fetching live detection: {e}')
        return jsonify({
            'success': True,
            'live_detection': detection_data,
            'recent_records': [],
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat()
        })

def _append_detection_log(session_id, source, message, details=None):
    """Helper function to append detection logs"""
    try:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'source': source,
            'message': message,
            'details': details or {}
        }
        app.logger.info(f'Detection log: {message}')
        return log_entry
    except Exception as e:
        app.logger.error(f'Failed to create detection log: {e}')
        return {}

@app.route('/api/session/<session_id>/upload-audio', methods=['POST'])
@login_required
def upload_session_audio(session_id):
    print("✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ audio recieved✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ")
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        return jsonify({'success': False, 'message': 'Session not found'}), 404
    
    sess = Session(sess_data)
    app.logger.info("upload-audio called by user_id=%s for session_id=%s", current_user.get_id(), session_id)

    if current_user.id != sess.learner_id:
        app.logger.warning("upload-audio denied for user_id=%s session_id=%s", current_user.get_id(), session_id)
        return jsonify({'success': False, 'message': 'Access denied'}), 403

    if 'audio' not in request.files:
        return jsonify({'success': False, 'message': 'No audio file provided'}), 400

    f = request.files['audio']
    audio_bytes = f.read()

    if not emotion_model:
        return jsonify({'success': False, 'message': 'Model not loaded'}), 500

    try:
        features, sr = _process_audio_bytes(audio_bytes)
        preds = emotion_model.predict(features, verbose=0)
        predicted_class = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))

        emotion_dict = {
            0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
            4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
        }
        label = emotion_dict.get(predicted_class, 'unknown')

        stress_emotions = {'angry': 0.3, 'fearful': 0.4, 'sad': 0.2, 'disgust': 0.15}
        emotion_stress = stress_emotions.get(label, 0.0)
        voice_stress_score = min(1.0, max(0.0, (1.0 - confidence) * 0.7 + emotion_stress))
        stress_label = 'low' if voice_stress_score < 0.35 else 'moderate' if voice_stress_score < 0.65 else 'high'

        suggestions = []
        if label in ('angry', 'fearful', 'sad'):
            suggestions.append('Learner showing negative emotion - offer reassurance and check understanding')
        if voice_stress_score > 0.6:
            suggestions.append('High voice stress detected - consider a short break or easier topic')
        if label == 'disgust':
            suggestions.append('Learner may be confused or frustrated - clarify the current topic')
        if confidence < 0.5:
            suggestions.append('Uncertain emotion detection - audio quality may be low')

        detection_payload = {
            'timestamp_utc': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'audio_emotion': {
                'primary': label,
                'confidence': round(confidence, 3),
                'scores': {emotion_dict[i]: round(float(preds[0][i]), 3) for i in range(len(preds[0]))}
            },
            'voice_stress': {
                'score': round(voice_stress_score, 3),
                'label': stress_label,
                'source': 'audio_model'
            },
            'video_emotion': {
                'status': 'model_pending',
                'note': 'Video emotion detection model not yet integrated'
            },
            'suggestions': suggestions
        }

        # Add a concise model log entry
        log_msg = f"Audio model: {label} (conf={confidence:.2f}) stress={voice_stress_score:.2f}"
        log_details = {
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'voice_stress_score': float(voice_stress_score),
            'scores': detection_payload['audio_emotion']['scores']
        }
        _appended_log = _append_detection_log(session_id, source='audio', message=log_msg, details=log_details)

        # ensure the cache's latest detection payload includes logs
        DETECTION_CACHE[session_id] = detection_payload
        DETECTION_CACHE[session_id].setdefault('logs', []).extend([_appended_log] if isinstance(_appended_log, dict) else [])

        app.logger.info('✓ ML Detection: session=%s | class=%d (%s) conf=%.2f | stress=%s (%.2f)', 
                       session_id, predicted_class, label, confidence, stress_label, voice_stress_score)
        print(f"[VOICE] Emotion: {label} (confidence: {confidence:.2f}, stress: {voice_stress_score:.2f}) - Session: {session_id}")

        # Store detection in Firestore - ALWAYS store, even if processing had issues
        session_id_str = str(session_id)
        try:
            detection_record = {
                'session_id': session_id_str,
                'timestamp': datetime.utcnow(),
                'detection_type': 'audio',
                'audio_emotion': detection_payload['audio_emotion'],
                'voice_stress': detection_payload['voice_stress'],
                'face_emotion': None,
                'gestures': [],
                'suggestions': detection_payload.get('suggestions', []),
                'created_at': datetime.utcnow().isoformat()
            }
            print(f"[DB] ⚡ STORING audio detection - Session ID: '{session_id_str}'")
            print(f"[DB] Emotion: {detection_payload['audio_emotion']['primary']}, Stress: {detection_payload['voice_stress']['score']}")
            
            # Firebase .add() returns (timestamp, document_reference) tuple
            write_result = db.collection('detection_records').add(detection_record)
            if isinstance(write_result, tuple) and len(write_result) == 2:
                timestamp, doc_ref = write_result
            else:
                doc_ref = write_result
            
            doc_id = doc_ref.id if hasattr(doc_ref, 'id') else str(doc_ref)
            app.logger.info(f'✓ Stored audio detection record for session {session_id_str}, doc_id: {doc_id}')
            print(f"[DB] ✅ SUCCESS: Audio detection stored - Session: {session_id_str}, Firebase Doc ID: {doc_id}")
            print(f"[DB] Stored data: session_id='{detection_record['session_id']}', type={detection_record['detection_type']}, emotion={detection_record['audio_emotion']['primary']}, stress={detection_record['voice_stress']['score']}")
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            app.logger.error(f'❌ CRITICAL: Failed to store audio detection record for session {session_id_str}: {e}')
            app.logger.error(f'Traceback: {error_details}')
            print(f"[ERROR] ❌ CRITICAL ERROR: Failed to store audio detection to Firebase!")
            print(f"[ERROR] Session ID: {session_id_str}")
            print(f"[ERROR] Error: {e}")
            print(f"[ERROR] Full traceback:\n{error_details}")
            # Don't fail the request, but log the error prominently

        room_name = f'session_{session_id}'
        app.logger.info('🔊 Emitting detection_update to room=%s namespace=/stream', room_name)
        print(f"[BROADCAST] Sending detection to trainers - Room: {room_name}")
        
        # Emit to both namespaces to ensure delivery
        socketio.emit('detection_update', detection_payload, room=room_name, namespace='/stream')
        socketio.emit('trainer_detection', detection_payload, room=room_name)
        
        app.logger.info('✓ Emitted detection_update successfully')
        print(f"[SUCCESS] Detection broadcast completed - Session: {session_id}")
        return jsonify({'success': True, 'detection': detection_payload})
    except Exception as e:
        app.logger.exception('Audio processing failed: %s', e)
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/session/<session_id>/upload-frame', methods=['POST'])
@login_required
def upload_session_frame(session_id):
    print("frames recieved")
    app.logger.info('upload-frame called by user_id=%s for session_id=%s', current_user.get_id(), session_id)
    
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        return jsonify({'success': False, 'message': 'Session not found'}), 404
    
    sess = Session(sess_data)
    
    if current_user.id != sess.learner_id:
        app.logger.warning('upload-frame denied for user_id=%s session_id=%s', current_user.get_id(), session_id)
        return jsonify({'success': False, 'message': 'Access denied'}), 403
    
    data = request.get_json()
    if not data or 'frame' not in data:
        app.logger.warning('No frame data in request')
        return jsonify({'success': False, 'message': 'No frame provided'}), 400
    
    LAST_FRAME_CACHE[session_id] = data['frame']
    
    frame_size = len(data['frame'])
    app.logger.info(f'Frame received: size={frame_size} bytes ({frame_size/1024:.1f}KB)')
    
    detect_gestures_from_frame.current_session_id = session_id
    
    gestures, face_emotion, face_confidence = detect_gestures_from_frame(data['frame'])
    
    if session_id not in DETECTION_CACHE:
        DETECTION_CACHE[session_id] = {
            'timestamp_utc': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'gestures': [],
            'suggestions': []
        }
    
    if session_id in DETECTION_CACHE:
        DETECTION_CACHE[session_id]['gestures'] = gestures
        DETECTION_CACHE[session_id]['timestamp_utc'] = datetime.utcnow().isoformat()
        
        DETECTION_CACHE[session_id]['face_emotion'] = {
            'emotion': face_emotion if face_emotion != 'not_detected' else 'Not Detected',
            'confidence': face_confidence,
            'detected': face_emotion != 'not_detected'
        }
        
        if 'Biting Nails' in gestures:
            if 'Learner showing nervous behavior (nail biting) - check stress level' not in DETECTION_CACHE[session_id].get('suggestions', []):
                DETECTION_CACHE[session_id].setdefault('suggestions', []).append('Learner showing nervous behavior (nail biting) - check stress level')
        if 'Hand on Jaw' in gestures:
            if 'Learner appears thoughtful or confused - may need clarification' not in DETECTION_CACHE[session_id].get('suggestions', []):
                DETECTION_CACHE[session_id].setdefault('suggestions', []).append('Learner appears thoughtful or confused - may need clarification')
        if 'Thumbs Up' in gestures:
            if 'Learner showing positive engagement!' not in DETECTION_CACHE[session_id].get('suggestions', []):
                DETECTION_CACHE[session_id].setdefault('suggestions', []).append('Learner showing positive engagement!')
        if 'Blink' in gestures:
            if 'Frequent blinking detected - may indicate fatigue or eye strain' not in DETECTION_CACHE[session_id].get('suggestions', []):
                DETECTION_CACHE[session_id].setdefault('suggestions', []).append('Frequent blinking detected - may indicate fatigue or eye strain')
        if 'Raising Hand' in gestures:
            if '✋ Learner is raising their hand - they may have a question!' not in DETECTION_CACHE[session_id].get('suggestions', []):
                DETECTION_CACHE[session_id].setdefault('suggestions', []).append('✋ Learner is raising their hand - they may have a question!')
        if 'Leaning Forward' in gestures:
            if 'Learner leaning in - showing strong focus and engagement' not in DETECTION_CACHE[session_id].get('suggestions', []):
                DETECTION_CACHE[session_id].setdefault('suggestions', []).append('Learner leaning in - showing strong focus and engagement')
        
        if face_emotion and face_emotion not in ['neutral', 'not_detected']:
            if face_emotion in ['angry', 'fear', 'sad']:
                suggestion = f'Face shows {face_emotion} emotion - learner may need support'
                if suggestion not in DETECTION_CACHE[session_id].get('suggestions', []):
                    DETECTION_CACHE[session_id].setdefault('suggestions', []).append(suggestion)
            elif face_emotion == 'happy':
                suggestion = 'Learner appears happy and engaged!'
                if suggestion not in DETECTION_CACHE[session_id].get('suggestions', []):
                    DETECTION_CACHE[session_id].setdefault('suggestions', []).append(suggestion)
        
        # Store video detection in Firestore - ALWAYS store, even if processing had issues
        session_id_str = str(session_id)
        try:
            face_emotion_data = DETECTION_CACHE[session_id].get('face_emotion') if session_id in DETECTION_CACHE else None
            detection_record = {
                'session_id': session_id_str,
                'timestamp': datetime.utcnow(),
                'detection_type': 'video',
                'audio_emotion': None,
                'voice_stress': None,
                'face_emotion': face_emotion_data,
                'gestures': gestures,
                'suggestions': DETECTION_CACHE[session_id].get('suggestions', []) if session_id in DETECTION_CACHE else [],
                'created_at': datetime.utcnow().isoformat()
            }
            print(f"[DB] ⚡ STORING video detection - Session ID: '{session_id_str}'")
            print(f"[DB] Gestures: {gestures}, Face emotion: {face_emotion_data}")
            
            # Firebase .add() returns (timestamp, document_reference) tuple
            write_result = db.collection('detection_records').add(detection_record)
            if isinstance(write_result, tuple) and len(write_result) == 2:
                timestamp, doc_ref = write_result
            else:
                doc_ref = write_result
            
            doc_id = doc_ref.id if hasattr(doc_ref, 'id') else str(doc_ref)
            log_msg = f'✓ Stored video detection record for session {session_id_str}, doc_id: {doc_id}'
            app.logger.info(log_msg)
            print(f"[DB] ✅ SUCCESS: Video detection stored - Session: {session_id_str}, Firebase Doc ID: {doc_id}")
            print(f"[DB] Stored data: session_id='{detection_record['session_id']}', type={detection_record['detection_type']}, gestures={len(gestures)}, face_emotion={face_emotion_data.get('emotion') if face_emotion_data else 'None'}")
            
            # Send terminal log to trainer interface
            room_name = f'session_{session_id}'
            socketio.emit('terminal_log', {'message': f'[DB] ✅ Video detection stored - Session: {session_id_str}, Doc ID: {doc_id}', 'type': 'info'}, room=room_name, namespace='/stream')
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f'❌ CRITICAL: Failed to store video detection record: {e}'
            app.logger.error(error_msg)
            app.logger.error(f'Traceback: {error_details}')
            print(f"[ERROR] ❌ CRITICAL ERROR: Failed to store video detection to Firebase!")
            print(f"[ERROR] Session ID: {session_id_str}")
            print(f"[ERROR] Error: {e}")
            print(f"[ERROR] Full traceback:\n{error_details}")
            room_name = f'session_{session_id}'
            socketio.emit('terminal_log', {'message': f'[ERROR] ❌ {error_msg}', 'type': 'error'}, room=room_name, namespace='/stream')
            # Don't fail the request, but log the error prominently
        
        room_name = f'session_{session_id}'
        emit_msg = f'🔊 Emitting detection_update to room={room_name} namespace=/stream'
        app.logger.info(emit_msg)
        socketio.emit('terminal_log', {'message': emit_msg, 'type': 'info'}, room=room_name, namespace='/stream')
        
        # Emit to both namespaces to ensure delivery
        socketio.emit('detection_update', DETECTION_CACHE[session_id], room=room_name, namespace='/stream')
        socketio.emit('trainer_detection', DETECTION_CACHE[session_id], room=room_name)
        
        success_msg = '✓ Emitted detection_update successfully'
        app.logger.info(success_msg)
        socketio.emit('terminal_log', {'message': success_msg, 'type': 'success'}, room=room_name, namespace='/stream')
    
    detection_summary = f'✓ Detection: session={session_id}, gestures={gestures}, face_emotion={face_emotion or "not_detected"} (conf={face_confidence:.2f})'
    app.logger.info(detection_summary)
    gesture_log = f"[GESTURE] Detected: {gestures}, Face: {face_emotion or 'not_detected'} - Session: {session_id}"
    print(gesture_log)
    
    # Send both logs to trainer interface
    room_name = f'session_{session_id}'
    socketio.emit('terminal_log', {'message': detection_summary, 'type': 'detection'}, room=room_name, namespace='/stream')
    socketio.emit('terminal_log', {'message': gesture_log, 'type': 'gesture'}, room=room_name, namespace='/stream')
    
    return jsonify({
        'success': True, 
        'gestures': gestures, 
        'face_emotion': face_emotion or 'not_detected', 
        'confidence': face_confidence
    })


# SocketIO handlers
@socketio.on('connect', namespace='/stream')
def socket_connect():
    app.logger.info('Socket client connected to /stream: %s', request.sid)
    emit('connect_response', {'data': 'Connected to detection stream'})
    print(f"[SOCKET] Trainer connected - Socket ID: {request.sid}")

@socketio.on('connect')
def socket_connect_default():
    app.logger.info('Socket client connected to default namespace: %s', request.sid)
    emit('connect_response', {'data': 'Connected to trainer detection stream'})

@socketio.on('join_session', namespace='/stream')
def socket_join_session(data):
    session_id = data.get('session_id')
    user_role = data.get('user_role', 'unknown')
    app.logger.info('join_session request: session_id=%s, socket_id=%s, role=%s', session_id, request.sid, user_role)
    
    if not session_id:
        app.logger.warning('join_session failed: no session_id provided')
        emit('error', {'message': 'session_id required'})
        return

    sess_data = get_session_by_id(str(session_id))
    if not sess_data:
        app.logger.warning('join_session failed: session %s not found', session_id)
        emit('error', {'message': 'Session not found'})
        return
    
    if request.sid is None:
        app.logger.warning('join_session failed: no socket ID')
        emit('error', {'message': 'No socket ID'})
        return

    room_name = f'session_{session_id}'
    join_room(room_name)
    app.logger.info('✓ Socket %s joined room %s successfully as %s', request.sid, room_name, user_role)
    
    # Send current detection data if trainer
    if user_role == 'trainer' and session_id in DETECTION_CACHE:
        emit('detection_update', DETECTION_CACHE[session_id])
        app.logger.info('✓ Sent current detection data to trainer')
    
    emit('joined_session', {'session_id': session_id, 'room': room_name, 'role': user_role})

@socketio.on('join_session')
def socket_join_session_default(data):
    """Handle join_session on default namespace for trainers"""
    session_id = data.get('session_id')
    user_role = data.get('user_role', 'trainer')
    app.logger.info('join_session (default) request: session_id=%s, socket_id=%s, role=%s', session_id, request.sid, user_role)
    
    if not session_id:
        emit('error', {'message': 'session_id required'})
        return

    room_name = f'session_{session_id}'
    join_room(room_name)
    app.logger.info('✓ Socket %s joined room %s (default namespace) as %s', request.sid, room_name, user_role)
    
    # Send current detection data if available
    if session_id in DETECTION_CACHE:
        emit('trainer_detection', DETECTION_CACHE[session_id])
        app.logger.info('✓ Sent current detection data to trainer (default namespace)')
    
    emit('joined_session', {'session_id': session_id, 'room': room_name, 'role': user_role})

@socketio.on('disconnect', namespace='/stream')
def socket_disconnect():
    app.logger.info('Socket client disconnected from /stream: %s', request.sid)

@socketio.on('disconnect')
def socket_disconnect_default():
    app.logger.info('Socket client disconnected from default namespace: %s', request.sid)

# Video session routes
@app.route('/session/<session_id>/video')
@login_required
def video_session(session_id):
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        flash('Session not found.', 'danger')
        return redirect(url_for('dashboard'))
    
    session = Session(sess_data)
    app.logger.info(f"video_session route accessed by user_id={current_user.get_id()} for session_id={session_id}")
    
    if current_user.id not in [session.trainer_id, session.learner_id]:
        flash('You do not have access to this session.', 'danger')
        return redirect(url_for('dashboard'))
    
    current_time = datetime.now()
    session_start = session.scheduled_time
    # Convert timezone-aware datetime to naive for comparison
    if session_start and hasattr(session_start, 'tzinfo') and session_start.tzinfo:
        session_start = session_start.replace(tzinfo=None)
    is_trainer = (current_user.id == session.trainer_id)
    can_join = is_trainer or (session_start and current_time >= (session_start - timedelta(minutes=15)))
    
    app.logger.info(
        "join-check user_id=%s session_id=%s server_now=%s session_start=%s is_trainer=%s can_join=%s status=%s",
        current_user.get_id(), session_id, current_time.isoformat(),
        session_start.isoformat() if session_start else 'None', is_trainer, can_join, session.status
    )
    
    if not can_join and session.status == 'scheduled':
        flash('Session is not available yet. Please join at the scheduled time.', 'warning')
        return redirect(url_for('dashboard'))
    
    if not session.video_room_name:
        session.video_room_name = generate_room_name()
        update_session(session_id, video_room_name=session.video_room_name)
        app.logger.info(f"Generated new video room for session_id={session.id}: {session.video_room_name}")
    
    user_display_name = f"{current_user.first_name} {current_user.last_name}"
    
    jitsi_config = {
        'room_name': session.video_room_name,
        'user_display_name': user_display_name,
        'user_role': 'trainer' if is_trainer else 'learner',
        'jitsi_server': get_jitsi_server_url(),
        'config_overwrite': {
            'startWithAudioMuted': False,
            'startWithVideoMuted': False,
            'startAudioOnly': False,
            'enableWelcomePage': False,
            'disableDeepLinking': True,
            'prejoinPageEnabled': False,
            'disableInviteFunctions': False,
            'requireDisplayName': True,
            'enableClosePage': True,
            'enableRecording': False,
            'liveStreamingEnabled': False,
            'hiddenPremeetingButtons': ['microphone', 'camera', 'select-background', 'invite'],
            'toolbarButtons': [
                'microphone', 'camera', 'closedcaptions', 'desktop', 'fullscreen',
                'fodeviceselection', 'hangup', 'profile', 'chat', 'recording',
                'livestreaming', 'settings', 'videoquality', 'filmstrip',
                'feedback', 'stats', 'shortcuts', 'tileview', 'videobackgroundblur',
                'help', 'mute-everyone', 'security'
            ],
            'constraints': {
                'video': {
                    'height': {
                        'ideal': 720,
                        'max': 720,
                        'min': 240
                    }
                }
            }
        },
        'interface_config_overwrite': {
            'SHOW_JITSI_WATERMARK': False,
            'SHOW_WATERMARK_FOR_GUESTS': False,
            'MOBILE_APP_PROMO': False,
            'VIDEO_QUALITY_LABEL_DISABLED': False,
            'SHOW_CHROME_EXTENSION_BANNER': False,
            'DEFAULT_BACKGROUND': '#474747',
            'DEFAULT_LOCAL_DISPLAY_NAME': user_display_name,
            'DEFAULT_REMOTE_DISPLAY_NAME': 'Participant',
            'ENABLE_DIAL_OUT': False,
            'ENABLE_FEEDBACK_ANIMATION': False,
            'GENERATE_ROOMNAMES_ON_WELCOME_PAGE': False,
            'INVITATION_POWERED_BY': False,
            'SHOW_BRAND_WATERMARK': False,
            'SHOW_POWERED_BY': False,
            'TOOLBAR_BUTTONS': [
                'microphone', 'camera', 'closedcaptions', 'desktop', 'fullscreen',
                'hangup', 'profile', 'chat', 'settings', 'videoquality', 'filmstrip',
                'feedback', 'stats', 'shortcuts', 'tileview', 'videobackgroundblur',
                'help', 'mute-everyone', 'security'
            ]
        },
        'connection_config': {
            'iceServers': [
                {'urls': 'stun:stun.l.google.com:19302'},
                {'urls': 'stun:stun1.l.google.com:19302'},
                {'urls': 'stun:stun2.l.google.com:19302'},
                {'urls': 'stun:stun3.l.google.com:19302'},
                {'urls': 'stun:stun4.l.google.com:19302'},
                {
                    'urls': 'turn:turn.anyfirewall.com:443?transport=tcp',
                    'username': 'webrtc',
                    'credential': 'webrtc'
                },
                {
                    'urls': 'turn:turn.bistri.com:80',
                    'username': 'homeo',
                    'credential': 'homeo'
                }
            ],
            'pcConfig': {
                'iceTransportPolicy': 'all',
                'bundlePolicy': 'max-bundle',
                'rtcpMuxPolicy': 'require'
            }
        },
        'detection': {
            'enabled': True,
            'api_endpoint': url_for('get_session_detection', session_id=session_id, _external=True),
            'recommended_poll_seconds': 4
        },
        'webrtc_options': {
            'disableSimulcast': False,
            'enableLayerSuspension': True,
            'p2p': {
                'enabled': True,
                'preferH264': True,
                'disableH264': False,
                'preferredCodec': 'H264'
            },
            'resolution': 720,
            'constraints': {
                'video': {
                    'height': {
                        'ideal': 720,
                        'max': 1080,
                        'min': 180
                    }
                }
            }
        }
    }
    
    app.logger.info(f"Rendering video template for session_id={session.id} room={session.video_room_name}")
    
    return render_template('video_session_jitsi.html', 
                         session=session,
                         jitsi_config=jitsi_config,
                         current_user=current_user)

@app.route('/api/session/<session_id>/start-video', methods=['POST'])
@login_required
def start_video_session(session_id):
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        return jsonify({'success': False, 'message': 'Session not found'})
    
    session = Session(sess_data)
    app.logger.info(f"start_video_session called by user_id={current_user.get_id()} for session_id={session_id}")
    
    if current_user.id != session.trainer_id:
        return jsonify({'success': False, 'message': 'Only trainers can start sessions'})
    
    if not session.video_room_name:
        session.video_room_name = generate_room_name()
        update_session(session_id, video_room_name=session.video_room_name)
    
    update_session(session_id, status='ongoing')
    app.logger.info(f"Session {session_id} marked ongoing, room={session.video_room_name}")
    
    return jsonify({
        'success': True,
        'room_name': session.video_room_name,
        'room_url': url_for('video_session', session_id=session_id, _external=True),
        'message': 'Video session started'
    })

@app.route('/api/session/<session_id>/end-video', methods=['POST'])
@login_required
def end_video_session(session_id):
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        return jsonify({'success': False, 'message': 'Session not found'})
    
    session = Session(sess_data)
    app.logger.info(f"end_video_session called by user_id={current_user.get_id()} for session_id={session_id}")
    
    if current_user.id != session.trainer_id:
        return jsonify({'success': False, 'message': 'Only trainers can end sessions'})
    
    update_session(session_id, status='completed', end_time=datetime.utcnow())
    app.logger.info(f"Session {session_id} marked completed")
    
    # Generate and send PDF report in background
    try:
        import threading
        
        def generate_report_async():
            try:
                pdf_path = generate_session_report(session_id)
                if pdf_path:
                    app.logger.info(f"✓ Session report generated: {pdf_path}")
                else:
                    app.logger.warning(f"Failed to generate report for session {session_id}")
            except Exception as e:
                app.logger.error(f"Error in async report generation: {e}")
        
        # Start report generation in a separate thread
        report_thread = threading.Thread(target=generate_report_async)
        report_thread.daemon = True
        report_thread.start()
        
    except Exception as e:
        app.logger.exception(f"Failed to start report generation: {e}")
    
    return jsonify({'success': True, 'message': 'Session ended successfully'})

@app.route('/api/session/<session_id>/get-video-room', methods=['GET'])
@login_required
def get_video_room(session_id):
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        return jsonify({'success': False, 'message': 'Session not found'})
    
    session = Session(sess_data)
    app.logger.info(f"get_video_room request by user_id={current_user.get_id()} for session_id={session_id}")
    
    if current_user.id not in [session.trainer_id, session.learner_id]:
        return jsonify({'success': False, 'message': 'Access denied'})
    
    if not session.video_room_name:
        return jsonify({'success': False, 'message': 'No video room created'})
    
    return jsonify({
        'success': True,
        'room_name': session.video_room_name,
        'room_url': f"{get_jitsi_server_url()}/{session.video_room_name}",
        'user_display_name': f"{current_user.first_name} {current_user.last_name}"
    })

@app.route('/session/<session_id>')
@login_required
def session_room(session_id):
    return redirect(url_for('video_session', session_id=session_id))

# Main routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user_data = get_user_by_email(form.email.data)
        if user_data and check_password_hash(user_data['password'], form.password.data):
            user = User(user_data)
            login_user(user, remember=form.remember_me.data)
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Login failed. Check your email and password.', 'danger')
    
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        existing_user = get_user_by_email(form.email.data)
        if existing_user:
            flash('Email already registered. Please use a different email.', 'danger')
            return render_template('register.html', form=form)
        
        hashed_password = generate_password_hash(form.password.data)
        user_id = create_user(
            email=form.email.data,
            password=hashed_password,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            user_type=form.user_type.data
        )
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.user_type == 'trainer':
        return redirect(url_for('trainer_dashboard'))
    else:
        return redirect(url_for('learner_dashboard'))

@app.route('/learner/dashboard')
@login_required
def learner_dashboard():
    if current_user.user_type != 'learner':
        flash('Access denied. This page is for learners only.', 'danger')
        return redirect(url_for('dashboard'))
    
    upcoming_sessions_data = get_sessions_by_learner(current_user.id, status='scheduled')
    now = datetime.now()
    upcoming_sessions = [Session(s) for s in upcoming_sessions_data if s.get('scheduled_time') and (s['scheduled_time'].replace(tzinfo=None) if hasattr(s['scheduled_time'], 'tzinfo') else s['scheduled_time']) >= now]
    upcoming_sessions.sort(key=lambda x: x.scheduled_time)
    
    completed_sessions_data = get_sessions_by_learner(current_user.id, status='completed')
    completed_sessions = [Session(s) for s in completed_sessions_data]
    completed_sessions.sort(key=lambda x: x.scheduled_time, reverse=True)
    
    total_hours = sum(session.duration for session in completed_sessions) / 60
    
    trainer_ids = {session.trainer_id for session in completed_sessions + upcoming_sessions}
    trainers_count = len(trainer_ids)

    detection_summary = {}
    
    return render_template('dashboard.html',
                         upcoming_sessions=upcoming_sessions,
                         completed_sessions=completed_sessions,
                         total_hours=round(total_hours, 1),
                         trainers_count=trainers_count,
                         detection_panel_enabled=True,
                         detection_summary=detection_summary)

@app.route('/trainer/dashboard')
@login_required
def trainer_dashboard():
    if current_user.user_type != 'trainer':
        flash('Access denied. This page is for trainers only.', 'danger')
        return redirect(url_for('dashboard'))
    
    total_sessions = count_sessions_by_trainer(current_user.id)
    
    upcoming_sessions_data = get_sessions_by_trainer(current_user.id, status='scheduled')
    now = datetime.now()
    upcoming_sessions_list = [Session(s) for s in upcoming_sessions_data if s.get('scheduled_time') and (s['scheduled_time'].replace(tzinfo=None) if hasattr(s['scheduled_time'], 'tzinfo') else s['scheduled_time']) >= now]
    upcoming_sessions_list.sort(key=lambda x: x.scheduled_time)
    
    upcoming_sessions_count = len(upcoming_sessions_list)
    
    completed_sessions_data = get_sessions_by_trainer(current_user.id, status='completed')
    completed_sessions = [Session(s) for s in completed_sessions_data]
    total_earnings = sum(session.price for session in completed_sessions)
    
    reviews_data = get_reviews_by_trainer(current_user.id)
    average_rating = sum(r['rating'] for r in reviews_data) / len(reviews_data) if reviews_data else 0
    
    stats = {
        'total_sessions': total_sessions,
        'upcoming_sessions': upcoming_sessions_count,
        'total_earnings': round(total_earnings, 2),
        'average_rating': round(average_rating, 1)
    }
    
    session_requests = []
    
    recent_reviews_data = get_reviews_by_trainer(current_user.id, limit=3)

    detection_overview = {}

    return render_template('trainer_dashboard.html',
                         stats=stats,
                         upcoming_sessions=upcoming_sessions_list,
                         session_requests=session_requests,
                         recent_reviews=recent_reviews_data,
                         detection_panel_enabled=True,
                         detection_overview=detection_overview)

@app.route('/session/create', methods=['GET', 'POST'])
@login_required
def create_session():
    if current_user.user_type != 'trainer':
        flash('Only trainers can create sessions.', 'danger')
        return redirect(url_for('dashboard'))
    
    form = SessionForm()
    if form.validate_on_submit():
        session_id = fb_create_session(
            title=form.title.data,
            description=form.description.data,
            category=form.category.data,
            scheduled_time=form.scheduled_time.data,
            duration=form.duration.data,
            trainer_id=current_user.id,
            price=float(form.price.data) if form.price.data else 0.0,
            max_participants=form.max_participants.data,
            difficulty=form.difficulty.data,
            prerequisites=form.prerequisites.data,
            materials=form.materials.data,
            is_recurring=form.is_recurring.data,
            recurrence_pattern=form.recurrence_pattern.data if form.is_recurring.data else None,
            recurrence_count=form.recurrence_count.data if form.is_recurring.data else 0
        )
        flash('Session created successfully!', 'success')
        return redirect(url_for('trainer_dashboard'))
    
    return render_template('create_session.html', form=form)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    form = ProfileForm()
    password_form = ChangePasswordForm()
    
    if form.validate_on_submit():
        update_data = {
            'first_name': form.first_name.data,
            'last_name': form.last_name.data,
            'email': form.email.data,
            'phone': form.phone.data,
            'bio': form.bio.data,
            'timezone': form.timezone.data
        }
        
        if current_user.user_type == 'trainer':
            update_data['specialization'] = form.specialization.data
            update_data['hourly_rate'] = float(form.hourly_rate.data) if form.hourly_rate.data else 0.0
            update_data['experience'] = form.experience.data
        
        update_user(current_user.id, **update_data)
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    if request.method == 'GET':
        form.first_name.data = current_user.first_name
        form.last_name.data = current_user.last_name
        form.email.data = current_user.email
        form.phone.data = current_user.phone
        form.bio.data = current_user.bio
        form.timezone.data = current_user.timezone
        
        if current_user.user_type == 'trainer':
            form.specialization.data = current_user.specialization
            form.hourly_rate.data = current_user.hourly_rate
            form.experience.data = current_user.experience
    
    return render_template('profile.html', form=form, password_form=password_form)

@app.route('/change-password', methods=['POST'])
@login_required
def change_password():
    password_form = ChangePasswordForm()
    form = ProfileForm()
    
    if password_form.validate_on_submit():
        if check_password_hash(current_user.password, password_form.current_password.data):
            update_user(current_user.id, password=generate_password_hash(password_form.new_password.data))
            flash('Password changed successfully!', 'success')
        else:
            flash('Current password is incorrect.', 'danger')
    else:
        for field, errors in password_form.errors.items():
            for error in errors:
                flash(f'{getattr(password_form, field).label.text}: {error}', 'danger')
    
    form.first_name.data = current_user.first_name
    form.last_name.data = current_user.last_name
    form.email.data = current_user.email
    form.phone.data = current_user.phone
    form.bio.data = current_user.bio
    form.timezone.data = current_user.timezone
    
    if current_user.user_type == 'trainer':
        form.specialization.data = current_user.specialization
        form.hourly_rate.data = current_user.hourly_rate
        form.experience.data = current_user.experience
    
    return render_template('profile.html', form=form, password_form=password_form)

@app.route('/browse-sessions')
@login_required
def browse_sessions():
    if current_user.user_type != 'learner':
        return redirect(url_for('dashboard'))
    
    available_sessions_data = get_available_sessions()
    app.logger.info(f'Found {len(available_sessions_data)} available sessions')
    
    now = datetime.now()
    available_sessions = []
    for s in available_sessions_data:
        if s.get('scheduled_time'):
            sched_time = s['scheduled_time'].replace(tzinfo=None) if hasattr(s['scheduled_time'], 'tzinfo') else s['scheduled_time']
            app.logger.info(f'Session {s.get("id")}: scheduled={sched_time}, now={now}, future={sched_time >= now}')
            if sched_time >= now:
                available_sessions.append(Session(s))
    
    available_sessions.sort(key=lambda x: x.scheduled_time)
    app.logger.info(f'Showing {len(available_sessions)} sessions to learner')
    
    return render_template('browse_sessions.html', sessions=available_sessions)

@app.route('/trainer/sessions')
@login_required
def trainer_sessions():
    if current_user.user_type != 'trainer':
        return redirect(url_for('dashboard'))
    
    sessions_data = get_sessions_by_trainer(current_user.id)
    sessions = [Session(s) for s in sessions_data]
    sessions.sort(key=lambda x: x.scheduled_time, reverse=True)
    
    return render_template('trainer_sessions.html', sessions=sessions)

@app.route('/api/session/<session_id>/book', methods=['POST'])
@login_required
def book_session(session_id):
    if current_user.user_type != 'learner':
        return jsonify({'success': False, 'message': 'Only learners can book sessions'})
    
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        return jsonify({'success': False, 'message': 'Session not found'})
    
    session = Session(sess_data)
    
    if session.learner_id is not None:
        return jsonify({'success': False, 'message': 'Session is already booked'})
    
    update_session(session_id, learner_id=current_user.id)
    
    return jsonify({'success': True, 'message': 'Session booked successfully'})

@app.route('/api/session/<session_id>/cancel', methods=['POST'])
@login_required
def cancel_session(session_id):
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        return jsonify({'success': False, 'message': 'Session not found'})
    
    session = Session(sess_data)
    
    if current_user.id not in [session.trainer_id, session.learner_id]:
        return jsonify({'success': False, 'message': 'Access denied'})
    
    update_session(session_id, status='cancelled', learner_id=None)
    
    return jsonify({'success': True, 'message': 'Session cancelled successfully'})

@app.route('/api/session/<session_id>/complete', methods=['POST'])
@login_required
def complete_session(session_id):
    if current_user.user_type != 'trainer':
        return jsonify({'success': False, 'message': 'Only trainers can complete sessions'})
    
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        return jsonify({'success': False, 'message': 'Session not found'})
    
    session = Session(sess_data)
    
    if session.trainer_id != current_user.id:
        return jsonify({'success': False, 'message': 'Access denied'})
    
    update_session(session_id, status='completed')
    
    return jsonify({'success': True, 'message': 'Session marked as completed'})

@app.route('/api/webrtc-config')
def webrtc_config():
    return jsonify({
        'success': True,
        'ice_servers': [
            {'urls': 'stun:stun.l.google.com:19302'},
            {'urls': 'stun:stun1.l.google.com:19302'},
            {'urls': 'stun:stun2.l.google.com:19302'},
            {'urls': 'stun:stun3.l.google.com:19302'},
            {'urls': 'stun:stun4.l.google.com:19302'},
        ],
        'jitsi_server': get_jitsi_server_url(),
        'requires_https': True
    })

@app.route('/api/media-devices')
def media_devices_info():
    return jsonify({
        'success': True,
        'requirements': {
            'video': True,
            'audio': True,
            'min_resolution': '640x480',
            'recommended_resolution': '1280x720',
            'codec_preference': 'H264',
            'browser_support': {
                'chrome': '>= 72',
                'firefox': '>= 68',
                'safari': '>= 12.1',
                'edge': '>= 79'
            }
        }
    })

@app.route('/api/check-jitsi')
def check_jitsi():
    is_reachable = check_jitsi_connectivity()
    return jsonify({
        'success': is_reachable,
        'jitsi_server': get_jitsi_server_url(),
        'reachable': is_reachable
    })

@app.route('/debug/network')
def network_debug():
    info = {
        'hostname': socket.gethostname(),
        'local_ip': socket.gethostbyname(socket.gethostname()),
        'server_time': datetime.utcnow().isoformat(),
        'jitsi_server': get_jitsi_server_url(),
        'jitsi_reachable': check_jitsi_connectivity(),
        'ports_to_check': {
            '5000': 'Flask app port (TCP)',
            '10000-20000': 'WebRTC media ports (UDP)',
            '443': 'HTTPS port'
        },
        'note': 'For local network streaming, use your local IP address, not localhost'
    }
    
    try:
        response = requests.get('https://api.ipify.org', timeout=3)
        info['external_ip'] = response.text
    except:
        info['external_ip'] = 'Unable to determine'
    
    return jsonify(info)

@app.route('/test/video')
@login_required
def test_video():
    return render_template('test_video.html')

@app.route('/test/socketio')
def test_socketio_page():
    return render_template('socketio_test.html')

@app.route('/trainer/reports')
@login_required
def session_reports():
    import sys
    print(f"\n[REPORTS] ===== Trainer Reports Page Accessed =====", flush=True)
    print(f"[REPORTS] User ID: {current_user.id}, User Type: {current_user.user_type}", flush=True)
    sys.stdout.flush()
    
    if current_user.user_type != 'trainer':
        flash('Access denied. This page is for trainers only.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Debug: Get ALL sessions first to see what statuses exist
    all_sessions_data = get_sessions_by_trainer(current_user.id)
    print(f"[REPORTS] Total sessions for trainer: {len(all_sessions_data)}")
    
    # Show status breakdown
    status_counts = {}
    for sess in all_sessions_data:
        status = sess.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"[REPORTS] Session status breakdown: {status_counts}")
    
    # Get completed sessions
    completed_sessions_data = get_sessions_by_trainer(current_user.id, status='completed')
    print(f"[REPORTS] Found {len(completed_sessions_data)} completed sessions")
    
    # Also check for sessions with detection records (even if not marked completed)
    # This helps find sessions that ended but weren't properly marked as completed
    print(f"[REPORTS] Checking for sessions with detection records...")
    sessions_with_detections = {}
    try:
        all_detections = db.collection('detection_records').stream()
        for doc in all_detections:
            record = doc.to_dict()
            sess_id = record.get('session_id')
            if sess_id:
                if sess_id not in sessions_with_detections:
                    sessions_with_detections[sess_id] = 0
                sessions_with_detections[sess_id] += 1
        
        print(f"[REPORTS] Found {len(sessions_with_detections)} unique sessions with detection records")
        if sessions_with_detections:
            print(f"[REPORTS] Sample session IDs with detections: {list(sessions_with_detections.keys())[:5]}")
    except Exception as e:
        print(f"[REPORTS] Error checking detection records: {e}")
    
    # Also include sessions that have detection records but might not be marked as completed
    # This is useful if sessions ended but weren't properly marked
    sessions_to_process = list(completed_sessions_data)
    
    # Add sessions with detections that aren't in completed list
    for sess_data in all_sessions_data:
        sess_id = str(sess_data.get('id'))
        if sess_id in sessions_with_detections and sess_id not in [str(s.get('id')) for s in completed_sessions_data]:
            print(f"[REPORTS] Found session {sess_id} with {sessions_with_detections[sess_id]} detections but status='{sess_data.get('status')}' - including it")
            sessions_to_process.append(sess_data)
    
    sessions = []
    
    for sess_data in sessions_to_process:
        # Create a dict with session data and additional info
        session_info = {
            'id': sess_data.get('id'),
            'title': sess_data.get('title'),
            'scheduled_time': sess_data.get('scheduled_time'),
            'duration': sess_data.get('duration'),
            'category': sess_data.get('category'),
            'learner_id': sess_data.get('learner_id'),
            'status': sess_data.get('status')
        }
        
        sess_id_str = str(session_info['id'])
        print(f"[REPORTS] Processing session: ID={sess_id_str}, Title={session_info.get('title', 'N/A')}, Status={session_info.get('status')}")
        
        # Use detection count from our earlier scan (more efficient)
        if sess_id_str in sessions_with_detections:
            session_info['detection_count'] = sessions_with_detections[sess_id_str]
            print(f"[REPORTS] ✓ Using cached detection count: {session_info['detection_count']} for session {sess_id_str}")
        else:
            # Fallback: query if not in cache (shouldn't happen, but just in case)
            try:
                detection_records = db.collection('detection_records')\
                    .where('session_id', '==', sess_id_str)\
                    .stream()
                session_info['detection_count'] = len(list(detection_records))
                print(f"[REPORTS] ✓ Queried detection count: {session_info['detection_count']} for session {sess_id_str}")
            except Exception as e:
                print(f"[REPORTS] ✗ ERROR fetching detection count for session {sess_id_str}: {e}")
                session_info['detection_count'] = 0
        
        # Get learner info
        if session_info['learner_id']:
            learner_data = get_user_by_id(session_info['learner_id'])
            if learner_data:
                session_info['learner'] = {
                    'first_name': learner_data.get('first_name', ''),
                    'last_name': learner_data.get('last_name', '')
                }
                print(f"[REPORTS] Learner: {session_info['learner']['first_name']} {session_info['learner']['last_name']}")
            else:
                session_info['learner'] = None
                print(f"[REPORTS] No learner data found for learner_id: {session_info['learner_id']}")
        else:
            session_info['learner'] = None
            print(f"[REPORTS] No learner_id for this session")
        
        sessions.append(session_info)
    
    # Sort by date, most recent first
    sessions.sort(key=lambda x: x['scheduled_time'] if x['scheduled_time'] else datetime.min, reverse=True)
    
    print(f"[REPORTS] Total sessions to display: {len(sessions)}")
    print(f"[REPORTS] Sessions with detections: {sum(1 for s in sessions if s.get('detection_count', 0) > 0)}")
    print(f"[REPORTS] ===== End Reports Processing =====\n")
    
    return render_template('session_reports.html', sessions=sessions)

@app.route('/session/<session_id>/report/download')
@login_required
def download_session_report(session_id):
    import sys
    print(f"\n[DOWNLOAD REPORT] ===== Download Report Request =====", flush=True)
    print(f"[DOWNLOAD REPORT] Session ID: {session_id}", flush=True)
    print(f"[DOWNLOAD REPORT] User ID: {current_user.id}", flush=True)
    sys.stdout.flush()
    
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        print(f"[DOWNLOAD REPORT] ✗ Session not found")
        flash('Session not found', 'danger')
        return redirect(url_for('dashboard'))
    
    session = Session(sess_data)
    print(f"[DOWNLOAD REPORT] Session found: Title={session.title}, Trainer ID={session.trainer_id}")
    
    if current_user.id != session.trainer_id:
        print(f"[DOWNLOAD REPORT] ✗ Access denied - User {current_user.id} is not trainer {session.trainer_id}")
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    # Generate report if not exists
    try:
        # Check if report already exists in session_reports folder
        import glob
        report_dir = 'session_reports'
        os.makedirs(report_dir, exist_ok=True)  # Ensure directory exists
        
        existing_reports = glob.glob(f'{report_dir}/session_{session_id}_*.pdf')
        print(f"[DOWNLOAD REPORT] Existing reports found: {len(existing_reports)}")
        
        if existing_reports:
            # Use the most recent report
            pdf_path = max(existing_reports, key=os.path.getctime)
            print(f"[DOWNLOAD REPORT] Using existing report: {pdf_path}")
            app.logger.info(f'Using existing report: {pdf_path}')
        else:
            # Generate new report
            print(f"[DOWNLOAD REPORT] No existing report found, generating new one...")
            app.logger.info(f'No existing report found, generating new one for session {session_id}')
            pdf_path = generate_session_report(session_id)
            print(f"[DOWNLOAD REPORT] Generated report path: {pdf_path}")
        
        if pdf_path and os.path.exists(pdf_path):
            from flask import send_file
            filename = f'session_report_{session_id}_{datetime.now().strftime("%Y%m%d")}.pdf'
            print(f"[DOWNLOAD REPORT] ✓ Sending file: {pdf_path} as {filename}")
            return send_file(pdf_path, as_attachment=True, download_name=filename)
        else:
            error_msg = f'PDF file not found at: {pdf_path}' if pdf_path else 'PDF generation returned None'
            print(f"[DOWNLOAD REPORT] ✗ {error_msg}")
            flash('Report generation failed - no PDF created', 'danger')
            app.logger.error(f'PDF generation failed for session {session_id}: {error_msg}')
            return redirect(url_for('session_reports'))
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[DOWNLOAD REPORT] ✗ EXCEPTION: {e}")
        print(f"[DOWNLOAD REPORT] Traceback:\n{error_trace}")
        flash(f'Report generation error: {str(e)}', 'danger')
        app.logger.error(f'PDF generation exception for session {session_id}: {e}')
        app.logger.error(error_trace)
        return redirect(url_for('session_reports'))

@app.route('/session/<session_id>/report/view')
@login_required
def view_session_report(session_id):
    import sys
    # Define session_id_str at the start to avoid UnboundLocalError
    session_id_str = str(session_id)
    
    print(f"\n[VIEW REPORT] ===== View Report Request =====", flush=True)
    print(f"[VIEW REPORT] Session ID: {session_id}", flush=True)
    print(f"[VIEW REPORT] User ID: {current_user.id}", flush=True)
    sys.stdout.flush()
    
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        print(f"[VIEW REPORT] ✗ Session not found")
        flash('Session not found', 'danger')
        return redirect(url_for('dashboard'))
    
    session = Session(sess_data)
    print(f"[VIEW REPORT] Session found: Title={session.title}, Trainer ID={session.trainer_id}")
    
    if current_user.id != session.trainer_id:
        print(f"[VIEW REPORT] ✗ Access denied - User {current_user.id} is not trainer {session.trainer_id}")
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get detection records - ensure session_id is a string to match how it's stored
    try:
        from google.cloud.firestore import Timestamp as FirestoreTimestamp
        print(f"[VIEW REPORT] Fetching detection records for session_id: '{session_id_str}' (type: {type(session_id_str).__name__})")
        app.logger.info(f'Fetching detection records for session_id: {session_id_str}')
        
        # Debug: Check what session_ids exist in detection_records
        try:
            all_detections_sample = db.collection('detection_records').limit(10).stream()
            sample_session_ids = set()
            for doc in all_detections_sample:
                record = doc.to_dict()
                sess_id = record.get('session_id')
                if sess_id:
                    sample_session_ids.add(str(sess_id))
            print(f"[VIEW REPORT] Sample session_ids found in detection_records: {list(sample_session_ids)[:5]}")
            if session_id_str in sample_session_ids:
                print(f"[VIEW REPORT] ✓ Session ID found in sample!")
            else:
                print(f"[VIEW REPORT] ⚠ Session ID NOT in sample - checking full query...")
        except Exception as debug_e:
            print(f"[VIEW REPORT] Debug check failed: {debug_e}")
        
        detection_records = db.collection('detection_records')\
            .where('session_id', '==', session_id_str)\
            .stream()
        
        records = []
        doc_count = 0
        for doc in detection_records:
            doc_count += 1
            record = doc.to_dict()
            print(f"[VIEW REPORT] Found record {doc_count}: session_id={record.get('session_id')}, type={record.get('detection_type')}")
            
            # Convert Firestore Timestamp to Python datetime if needed
            if 'timestamp' in record and record['timestamp']:
                if isinstance(record['timestamp'], FirestoreTimestamp):
                    record['timestamp'] = record['timestamp'].to_datetime()
                elif isinstance(record['timestamp'], datetime):
                    # Already a datetime, keep it
                    pass
                else:
                    # Try to parse if it's a string
                    try:
                        if isinstance(record['timestamp'], str):
                            record['timestamp'] = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                    except:
                        app.logger.warning(f'Could not parse timestamp: {record.get("timestamp")}')
            
            records.append(record)
        
        print(f"[VIEW REPORT] Found {len(records)} detection records (queried {doc_count} docs)")
        app.logger.info(f'Found {len(records)} detection records for session {session_id_str}')
        
        if len(records) == 0:
            print(f"[VIEW REPORT] ⚠ No records found! Checking if ANY detection_records exist...")
            try:
                total_count = len(list(db.collection('detection_records').limit(100).stream()))
                print(f"[VIEW REPORT] Total detection_records in database (sample): {total_count}")
            except Exception as e:
                print(f"[VIEW REPORT] Error checking total count: {e}")
        if records:
            print(f"[VIEW REPORT] Sample record keys: {list(records[0].keys())}")
            app.logger.info(f'Sample record keys: {list(records[0].keys())}')
            app.logger.info(f'Sample record: {records[0]}')
        else:
            print(f"[VIEW REPORT] ⚠ No detection records found for this session")
        
        # Sort by timestamp in Python
        records.sort(key=lambda x: x.get('timestamp', datetime.min) if x.get('timestamp') else datetime.min)
        
        # Calculate statistics for the template
        audio_count = sum(1 for r in records if r.get('detection_type') == 'audio')
        video_count = sum(1 for r in records if r.get('detection_type') == 'video')
        stress_scores = [r.get('voice_stress', {}).get('score', 0) for r in records 
                        if r.get('voice_stress') and r.get('voice_stress', {}).get('score')]
        avg_stress = (sum(stress_scores) / len(stress_scores) * 100) if stress_scores else None
        
        stats = {
            'total': len(records),
            'audio': audio_count,
            'video': video_count,
            'avg_stress': avg_stress
        }
        print(f"[VIEW REPORT] Stats calculated: {stats}")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[VIEW REPORT] ✗ EXCEPTION fetching records: {e}")
        print(f"[VIEW REPORT] Traceback:\n{error_trace}")
        app.logger.error(f'Error fetching detection records for session {session_id}: {e}')
        app.logger.error(error_trace)
        records = []
        stats = {
            'total': 0,
            'audio': 0,
            'video': 0,
            'avg_stress': None
        }
    
    # Get learner info - pass as dict for template compatibility
    learner = None
    if session.learner_id:
        learner_data = get_user_by_id(session.learner_id)
        if learner_data:
            learner = learner_data  # Pass as dict, template can access with .get() or ['key']
            print(f"[VIEW REPORT] Learner: {learner_data.get('first_name')} {learner_data.get('last_name')}")
        else:
            print(f"[VIEW REPORT] ⚠ Learner ID {session.learner_id} not found")
    else:
        print(f"[VIEW REPORT] No learner_id for this session")
    
    # Debug: Log what we're sending to template
    print(f"[VIEW REPORT] Rendering template with {len(records)} records, stats={stats}")
    app.logger.info(f'Rendering report for session {session_id_str}: {len(records)} records, stats={stats}')
    
    # Ensure scheduled_time is a datetime object for template
    if session.scheduled_time and not isinstance(session.scheduled_time, datetime):
        try:
            from google.cloud.firestore import Timestamp as FirestoreTimestamp
            if isinstance(session.scheduled_time, FirestoreTimestamp):
                session.scheduled_time = session.scheduled_time.to_datetime()
        except:
            pass
    
    try:
        print(f"[VIEW REPORT] ===== End View Report =====\n")
        return render_template('view_report.html', session=session, records=records, learner=learner, stats=stats)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[VIEW REPORT] ✗ TEMPLATE RENDERING ERROR: {e}")
        print(f"[VIEW REPORT] Traceback:\n{error_trace}")
        app.logger.error(f'Template rendering error for session {session_id_str}: {e}')
        app.logger.error(error_trace)
        flash(f'Error loading report: {str(e)}', 'danger')
        return redirect(url_for('session_reports'))

@app.route('/test/reports')
def test_reports_route():
    """Simple test route to verify server is responding"""
    import sys
    print("=" * 50, flush=True)
    print("[TEST] Test reports route accessed!", flush=True)
    print("=" * 50, flush=True)
    sys.stdout.flush()
    return jsonify({
        'status': 'success',
        'message': 'Server is responding correctly',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/test/create-detection/<session_id>')
@login_required
def test_create_detection(session_id):
    """Test endpoint to create a sample detection record"""
    try:
        session_id_str = str(session_id)
        
        # Verify session exists
        sess_data = get_session_by_id(session_id)
        if not sess_data:
            return jsonify({'error': 'Session not found'}), 404
        
        session = Session(sess_data)
        if current_user.id != session.trainer_id:
            return jsonify({'error': 'Access denied'}), 403
        
        # Create a test detection record
        test_record = {
            'session_id': session_id_str,
            'timestamp': datetime.utcnow(),
            'detection_type': 'audio',
            'audio_emotion': {
                'primary': 'happy',
                'confidence': 0.85,
                'scores': {'happy': 0.85, 'neutral': 0.10, 'sad': 0.05}
            },
            'voice_stress': {
                'score': 0.25,
                'label': 'low',
                'source': 'test'
            },
            'face_emotion': None,
            'gestures': [],
            'suggestions': ['Test detection record'],
            'created_at': datetime.utcnow().isoformat(),
            'test_record': True
        }
        
        print(f"[TEST] Creating test detection record for session: {session_id_str}")
        write_result = db.collection('detection_records').add(test_record)
        # Firebase .add() returns (timestamp, document_reference) tuple
        if isinstance(write_result, tuple) and len(write_result) == 2:
            timestamp, doc_ref = write_result
        else:
            doc_ref = write_result
        
        doc_id = doc_ref.id if hasattr(doc_ref, 'id') else str(doc_ref)
        print(f"[TEST] ✓ Test detection record created - Doc ID: {doc_id}")
        
        return jsonify({
            'success': True,
            'message': 'Test detection record created',
            'doc_id': doc_id,
            'session_id': session_id_str,
            'record': test_record
        })
    except Exception as e:
        import traceback
        print(f"[TEST] ✗ Error creating test detection: {e}")
        print(f"[TEST] Traceback:\n{traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/debug/session/<session_id>/detections')
@login_required
def debug_session_detections(session_id):
    """Debug endpoint to check detection records"""
    sess_data = get_session_by_id(session_id)
    if not sess_data:
        return jsonify({'error': 'Session not found'}), 404
    
    session = Session(sess_data)
    if current_user.id != session.trainer_id:
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        from google.cloud.firestore import Timestamp as FirestoreTimestamp
        
        session_id_str = str(session_id)
        detection_records = db.collection('detection_records')\
            .where('session_id', '==', session_id_str)\
            .stream()
        
        records = []
        for doc in detection_records:
            record = doc.to_dict()
            record['_doc_id'] = doc.id
            
            # Convert timestamp for JSON serialization
            if 'timestamp' in record and record['timestamp']:
                if isinstance(record['timestamp'], FirestoreTimestamp):
                    record['timestamp'] = record['timestamp'].to_datetime().isoformat()
                elif isinstance(record['timestamp'], datetime):
                    record['timestamp'] = record['timestamp'].isoformat()
            
            records.append(record)
        
        return jsonify({
            'session_id': session_id_str,
            'count': len(records),
            'records': records
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'database': 'firebase',
        'jitsi': check_jitsi_connectivity()
    })

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.exception('Internal server error: %s', error)
    return render_template('500.html'), 500

def create_ssl_certificates():
    cert_file = 'cert.pem'
    key_file = 'key.pem'
    
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        print("Generating self-signed certificates for HTTPS...")
        
        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 2048)
        
        cert = crypto.X509()
        cert.get_subject().C = "US"
        cert.get_subject().ST = "State"
        cert.get_subject().L = "City"
        cert.get_subject().O = "DirectProf"
        cert.get_subject().OU = "Development"
        cert.get_subject().CN = socket.gethostname()
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(365*24*60*60)
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(k)
        cert.sign(k, 'sha256')
        
        with open(cert_file, "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        with open(key_file, "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
        
        print(f"Certificates created: {cert_file}, {key_file}")
    
    return (cert_file, key_file)

def get_network_info():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n{'='*60}")
    print(f"DirectProf - Firebase Edition")
    print(f"{'='*60}")
    print(f"Hostname: {hostname}")
    print(f"Local IP: {local_ip}")
    print(f"Server URL: https://{local_ip}:5000")
    print(f"Local URL: https://localhost:5000")
    print(f"Jitsi Server: {get_jitsi_server_url()}")
    print(f"Jitsi Reachable: {check_jitsi_connectivity()}")
    print(f"Database: Firebase Firestore")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    cert_paths = create_ssl_certificates()
    get_network_info()
    
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=True,
        ssl_context=cert_paths,
        allow_unsafe_werkzeug=True
    )


