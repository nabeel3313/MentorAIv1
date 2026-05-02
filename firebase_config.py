"""
Firebase Configuration and Helper Functions
Replace SQLAlchemy with Firebase Firestore
"""

import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Firebase
cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH', 'firebase-credentials.json')
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

# Collections
USERS_COLLECTION = 'users'
SESSIONS_COLLECTION = 'sessions'
BOOKINGS_COLLECTION = 'bookings'
REVIEWS_COLLECTION = 'reviews'

# User Class for Flask-Login
class User:
    def __init__(self, user_data):
        self.id = user_data.get('id')
        self.email = user_data.get('email')
        self.password = user_data.get('password')
        self.first_name = user_data.get('first_name')
        self.last_name = user_data.get('last_name')
        self.user_type = user_data.get('user_type')
        self.phone = user_data.get('phone')
        self.bio = user_data.get('bio')
        self.specialization = user_data.get('specialization')
        self.hourly_rate = user_data.get('hourly_rate', 0.0)
        self.experience = user_data.get('experience', 0)
        self.timezone = user_data.get('timezone', 'UTC')
        self.created_at = user_data.get('created_at')
        self.updated_at = user_data.get('updated_at')
    
    def is_authenticated(self):
        return True
    
    def is_active(self):
        return True
    
    def is_anonymous(self):
        return False
    
    def get_id(self):
        return str(self.id)

# Session Class
class Session:
    def __init__(self, session_data):
        self.id = session_data.get('id')
        self.title = session_data.get('title')
        self.description = session_data.get('description')
        self.category = session_data.get('category')
        self.scheduled_time = session_data.get('scheduled_time')
        self.duration = session_data.get('duration')
        self.price = session_data.get('price', 0.0)
        self.max_participants = session_data.get('max_participants', 1)
        self.difficulty = session_data.get('difficulty', 'beginner')
        self.prerequisites = session_data.get('prerequisites')
        self.materials = session_data.get('materials')
        self.is_recurring = session_data.get('is_recurring', False)
        self.recurrence_pattern = session_data.get('recurrence_pattern')
        self.recurrence_count = session_data.get('recurrence_count', 0)
        self.status = session_data.get('status', 'scheduled')
        self.trainer_id = session_data.get('trainer_id')
        self.learner_id = session_data.get('learner_id')
        self.video_room_name = session_data.get('video_room_name')
        self.created_at = session_data.get('created_at')
        self._trainer = None
        self._learner = None
    
    @property
    def trainer(self):
        if not self._trainer and self.trainer_id:
            user_data = get_user_by_id(self.trainer_id)
            if user_data:
                self._trainer = User(user_data)
        return self._trainer
    
    @property
    def learner(self):
        if not self._learner and self.learner_id:
            user_data = get_user_by_id(self.learner_id)
            if user_data:
                self._learner = User(user_data)
        return self._learner

# Helper Functions
def create_user(email, password, first_name, last_name, user_type, **kwargs):
    user_ref = db.collection(USERS_COLLECTION).document()
    user_data = {
        'email': email,
        'password': password,
        'first_name': first_name,
        'last_name': last_name,
        'user_type': user_type,
        'phone': kwargs.get('phone'),
        'bio': kwargs.get('bio'),
        'specialization': kwargs.get('specialization'),
        'hourly_rate': kwargs.get('hourly_rate', 0.0),
        'experience': kwargs.get('experience', 0),
        'timezone': kwargs.get('timezone', 'UTC'),
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }
    user_ref.set(user_data)
    return user_ref.id

def get_user_by_id(user_id):
    doc = db.collection(USERS_COLLECTION).document(user_id).get()
    if doc.exists:
        data = doc.to_dict()
        data['id'] = doc.id
        return data
    return None

def get_user_by_email(email):
    users = db.collection(USERS_COLLECTION).where('email', '==', email).limit(1).stream()
    for user in users:
        data = user.to_dict()
        data['id'] = user.id
        return data
    return None

def update_user(user_id, **kwargs):
    kwargs['updated_at'] = datetime.utcnow()
    db.collection(USERS_COLLECTION).document(user_id).update(kwargs)

def create_session(title, description, category, scheduled_time, duration, trainer_id, **kwargs):
    session_ref = db.collection(SESSIONS_COLLECTION).document()
    session_data = {
        'title': title,
        'description': description,
        'category': category,
        'scheduled_time': scheduled_time,
        'duration': duration,
        'price': kwargs.get('price', 0.0),
        'max_participants': kwargs.get('max_participants', 1),
        'difficulty': kwargs.get('difficulty', 'beginner'),
        'prerequisites': kwargs.get('prerequisites'),
        'materials': kwargs.get('materials'),
        'is_recurring': kwargs.get('is_recurring', False),
        'recurrence_pattern': kwargs.get('recurrence_pattern'),
        'recurrence_count': kwargs.get('recurrence_count', 0),
        'status': kwargs.get('status', 'scheduled'),
        'trainer_id': trainer_id,
        'learner_id': kwargs.get('learner_id'),
        'video_room_name': kwargs.get('video_room_name'),
        'created_at': datetime.utcnow()
    }
    session_ref.set(session_data)
    return session_ref.id

def get_session_by_id(session_id):
    doc = db.collection(SESSIONS_COLLECTION).document(session_id).get()
    if doc.exists:
        data = doc.to_dict()
        data['id'] = doc.id
        return data
    return None

def update_session(session_id, **kwargs):
    db.collection(SESSIONS_COLLECTION).document(session_id).update(kwargs)

def delete_session(session_id):
    db.collection(SESSIONS_COLLECTION).document(session_id).delete()

def get_sessions_by_trainer(trainer_id, status=None, limit=None):
    query = db.collection(SESSIONS_COLLECTION).where('trainer_id', '==', trainer_id)
    if status:
        query = query.where('status', '==', status)
    if limit:
        query = query.limit(limit)
    sessions = []
    for doc in query.stream():
        data = doc.to_dict()
        data['id'] = doc.id
        sessions.append(data)
    return sessions

def get_sessions_by_learner(learner_id, status=None, limit=None):
    query = db.collection(SESSIONS_COLLECTION).where('learner_id', '==', learner_id)
    if status:
        query = query.where('status', '==', status)
    if limit:
        query = query.limit(limit)
    sessions = []
    for doc in query.stream():
        data = doc.to_dict()
        data['id'] = doc.id
        sessions.append(data)
    return sessions

def get_available_sessions(limit=None):
    # Firestore doesn't support querying for None/null values with ==
    # Get all scheduled sessions and filter in Python
    query = db.collection(SESSIONS_COLLECTION).where('status', '==', 'scheduled')
    if limit:
        query = query.limit(limit * 2)  # Get more to account for filtering
    sessions = []
    for doc in query.stream():
        data = doc.to_dict()
        data['id'] = doc.id
        # Only include sessions without a learner
        if not data.get('learner_id'):
            sessions.append(data)
            if limit and len(sessions) >= limit:
                break
    return sessions

def count_sessions_by_trainer(trainer_id):
    sessions = db.collection(SESSIONS_COLLECTION).where('trainer_id', '==', trainer_id).stream()
    return sum(1 for _ in sessions)

def create_booking(session_id, learner_id, status='pending'):
    booking_ref = db.collection(BOOKINGS_COLLECTION).document()
    booking_data = {
        'session_id': session_id,
        'learner_id': learner_id,
        'status': status,
        'booked_at': datetime.utcnow()
    }
    booking_ref.set(booking_data)
    return booking_ref.id

def get_bookings_by_status(status, trainer_id=None):
    query = db.collection(BOOKINGS_COLLECTION).where('status', '==', status)
    bookings = []
    for doc in query.stream():
        data = doc.to_dict()
        data['id'] = doc.id
        if trainer_id:
            session_data = get_session_by_id(data['session_id'])
            if session_data and session_data.get('trainer_id') == trainer_id:
                bookings.append(data)
        else:
            bookings.append(data)
    return bookings

def create_review(session_id, learner_id, trainer_id, rating, comment):
    review_ref = db.collection(REVIEWS_COLLECTION).document()
    review_data = {
        'session_id': session_id,
        'learner_id': learner_id,
        'trainer_id': trainer_id,
        'rating': rating,
        'comment': comment,
        'created_at': datetime.utcnow()
    }
    review_ref.set(review_data)
    return review_ref.id

def get_reviews_by_trainer(trainer_id, limit=None):
    query = db.collection(REVIEWS_COLLECTION).where('trainer_id', '==', trainer_id)
    if limit:
        query = query.limit(limit)
    reviews = []
    for doc in query.stream():
        data = doc.to_dict()
        data['id'] = doc.id
        reviews.append(data)
    # Sort in Python instead of Firestore to avoid index requirement
    reviews.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
    return reviews

def count_users():
    users = db.collection(USERS_COLLECTION).stream()
    return sum(1 for _ in users)
