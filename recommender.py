import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

class ExerciseRecommender:
    def __init__(self):
        self.model = None
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Загружаем существующую модель или обучаем новую"""
        model_path = 'exercise_model.pkl'
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.train_model()
            joblib.dump(self.model, model_path)
    
    def train_model(self):
        """Обучаем модель на синтетических данных"""
        np.random.seed(42)
        n_samples = 10000
        
        # Генерация данных
        bmi = np.random.normal(24, 4, n_samples)
        hour = np.random.randint(6, 23, n_samples)
        posture_risk = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
        recovery_score = np.random.randint(0, 4, n_samples)
        age = np.random.randint(18, 65, n_samples)
        
        # Логика генерации целевой переменной
        def get_target(row):
            bmi_val, hour_val, posture, recovery, age_val = row
            
            # Возрастные ограничения
            if age_val > 60:
                return 1 if recovery < 3 else 0
            
            # Проблемы с осанкой
            if posture in [1, 2]:
                if recovery >= 3:
                    return 0
                elif 6 <= hour_val <= 12:
                    return 1
                else:
                    return 2
            else:
                if recovery >= 3:
                    return 1
                elif recovery <= 1 and (6 <= hour_val <= 11):
                    return 3
                else:
                    return 2
        
        # Создаем датасет
        X = np.column_stack([bmi, hour, posture_risk, recovery_score, age])
        y = np.apply_along_axis(get_target, 1, X)
        
        # Обучаем модель
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        self.model.fit(X, y)
    
    def get_recommendation(self, bmi, hour, posture_risk, recovery_score, age, height, weight):
        """Получаем рекомендацию для пользователя"""
        
        # Подготовка данных для модели
        features = np.array([[bmi, hour, posture_risk, recovery_score, age]])
        
        # Предсказание
        pred = self.model.predict(features)[0]
        
        # Формируем текстовую рекомендацию с деталями
        recommendations = {
            0: {
                'type': 0,
                'title': '🚨 Отдых и ЛФК',
                'text': f"""На основе анализа ваших данных (BMI: {bmi:.1f}, осанка: {self._posture_text(posture_risk)}, 
                восстановление: {recovery_score}/3) рекомендуем сегодня воздержаться от интенсивных тренировок.
                
                📋 Рекомендации:
                • Лечебная физкультура для коррекции осанки
                • Легкая растяжка (15-20 минут)
                • Массаж спины с роликом
                • Прогулка на свежем воздухе (30-40 минут)
                
                ⚠️ Противопоказано: бег, приседания со штангой, становая тяга.""",
                'exercises': ['Растяжка грудного отдела', 'Упражнения на стабилизацию корпуса', 'Плавание (при возможности)']
            },
            1: {
                'type': 1,
                'title': '🧘 Легкая активность',
                'text': f"""Ваше состояние позволяет выполнять легкие физические нагрузки. 
                BMI: {bmi:.1f} | Время тренировки: {hour}:00 | Осанка: {self._posture_text(posture_risk)}
                
                📋 Рекомендации:
                • Йога или пилатес (30-40 минут)
                • Легкая кардио: ходьба 5-7 км/ч (20-30 минут)
                • Растяжка всех групп мышц
                • Техническая работа над движениями
                
                🎯 Фокус: восстановление, техника, мобильность.""",
                'exercises': ['Кошка-корова', 'Поза ребенка', 'Ходьба на беговой дорожке']
            },
            2: {
                'type': 2,
                'title': '🏋️‍♂️ Силовая без осевой нагрузки',
                'text': f"""Рекомендуется силовая тренировка с исключением осевых нагрузок на позвоночник.
                
                📋 Программа (45-60 минут):
                1. Разминка: суставная гимнастика (10 мин)
                2. Жим гантелей сидя - 3x12
                3. Тяга верхнего блока - 3x12
                4. Жим ногами в тренажере - 3x15
                5. Сгибание рук с гантелями - 3x12
                6. Гиперэкстензия - 3x15
                7. Заминка: растяжка (10 мин)
                
                ⚠️ Исключить: приседания, становую тягу, жим стоя.
                🎯 Интенсивность: 70-80% от максимума.""",
                'exercises': ['Жим гантелей сидя', 'Тяга блока', 'Жим ногами']
            },
            3: {
                'type': 3,
                'title': '💪 Полноценная тренировка',
                'text': f"""Отличное состояние! Вы готовы к полноценной высокоинтенсивной тренировке.
                
                📋 Рекомендуемая программа (60-90 минут):
                • Разминка: 10-15 минут
                • Базовые упражнения:
                  - Приседания со штангой 4x8-10
                  - Жим лежа 4x8-10
                  - Тяга штанги в наклоне 4x10
                • Добивка: изолирующие упражнения
                • Кардио: 15-20 минут
                • Заминка и растяжка
                
                🎯 Интенсивность: 80-90% от максимума
                💪 Фокус: развитие силы и мышечной массы.
                
                ⏰ Лучшее время для такой нагрузки - утренние часы (6:00-11:00), 
                когда уровень кортизола наиболее подходит для силовой работы.""",
                'exercises': ['Приседания со штангой', 'Жим лежа', 'Становая тяга']
            }
        }
        
        rec = recommendations[pred]
        
        # Добавляем персонализированные советы
        if bmi < 18.5:
            rec['text'] += "\n\n⚠️ У вас недостаточный вес. Обратите особое внимание на питание: увеличьте калорийность рациона, добавьте больше белка."
        elif bmi > 30:
            rec['text'] += "\n\n⚠️ Избыточный вес. Начинайте с низкоинтенсивных нагрузок, следите за пульсом. Рекомендуется плавание и ходьба."
        
        if hour < 8:
            rec['text'] += "\n\n🌅 Утренняя тренировка: хорошо разомнитесь, так как утром мышцы более жесткие."
        elif hour > 20:
            rec['text'] += "\n\n🌙 Вечерняя тренировка: не используйте слишком стимулирующие упражнения за 2-3 часа до сна."
        
        return rec
    
    def _posture_text(self, posture_risk):
        """Текстовое описание состояния осанки"""
        if posture_risk == 0:
            return "хорошая"
        elif posture_risk == 1:
            return "сутулость (гиперкифоз)"
        else:
            return "асимметрия плеч"

# Для совместимости со старым кодом
def get_recommendation(bmi, hour, posture_risk, recovery_score):
    """Старая функция для совместимости"""
    recommender = ExerciseRecommender()
    return recommender.get_recommendation(bmi, hour, posture_risk, recovery_score, age=30, height=170, weight=70)