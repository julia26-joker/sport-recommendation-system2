from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    recommendations = db.relationship('RecommendationHistory', backref='user', lazy=True)

class RecommendationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Параметры пользователя
    height = db.Column(db.Float)
    weight = db.Column(db.Float)
    age = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    training_hour = db.Column(db.Integer)
    recovery_score = db.Column(db.Integer)
    posture_risk = db.Column(db.Integer)
    
    # Результат
    recommendation_text = db.Column(db.Text)
    recommendation_type = db.Column(db.Integer)  # 0,1,2,3
    
    # Метаданные
    photo_path = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)