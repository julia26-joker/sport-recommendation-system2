import cv2
import numpy as np
import os
from datetime import datetime

# Новый API MediaPipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PostureAnalyzer:
    def __init__(self):
        """Инициализация ML-модели для определения позы"""
        # Путь к скачанной ML-модели
        model_path = 'pose_landmarker.task'
        
        if not os.path.exists(model_path):
            print(f"Ошибка: Модель не найдена по пути {model_path}")
            self.landmarker = None
            return
        
        # Настройка ML-модели для работы с изображениями
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,  # Режим для статичных фото
            num_poses=1,  # Определяем одного человека
            min_pose_detection_confidence=0.5  # Минимальная уверенность модели 50%
        )
        
        # Загружаем ML-модель
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        print("✅ ML-модель для анализа осанки загружена")
    
    def analyze(self, image_path, output_path=None):
        """
        Анализ позы на фото с помощью ML-модели MediaPipe
        Возвращает: (posture_risk, result_text, output_image_path)
        """
        if self.landmarker is None:
            return 0, "Модель ML не загружена", image_path
        
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            return 0, "Ошибка: не удалось загрузить фото", None
        
        height, width = image.shape[:2]
        
        # Конвертируем в RGB и в формат MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # ML-детекция: находим ключевые точки
        detection_result = self.landmarker.detect(mp_image)
        
        # Проверяем, нашла ли ML-модель человека
        if not detection_result.pose_landmarks:
            return 0, "ML-модель не обнаружила человека. Сделайте фото в полный рост", image_path
        
        # Берем первого найденного человека
        pose_landmarks = detection_result.pose_landmarks[0]
        
        # Рисуем ключевые точки на изображении (красные метки)
        annotated_image = image.copy()
        
        # Рисуем все ключевые точки красными кружками
        for landmark in pose_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)  # Красный кружок
        
        # Рисуем соединения между точками (скелет) для наглядности
        # Индексы соединений
        connections = [
            (11, 12),  # плечи
            (11, 13), (13, 15),  # левая рука
            (12, 14), (14, 16),  # правая рука
            (11, 23), (12, 24),  # корпус
            (23, 24),  # таз
            (23, 25), (25, 27),  # левая нога
            (24, 26), (26, 28),  # правая нога
            (11, 12)  # плечи
        ]
        
        for connection in connections:
            idx1, idx2 = connection
            x1 = int(pose_landmarks[idx1].x * width)
            y1 = int(pose_landmarks[idx1].y * height)
            x2 = int(pose_landmarks[idx2].x * width)
            y2 = int(pose_landmarks[idx2].y * height)
            cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зеленые линии
        
        # --- Анализ осанки: вычисляем различия между плечами ---
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        
        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
        
        # --- Выносим вердикт ---
        if shoulder_diff > 0.04:  # Разница больше 4% от высоты кадра
            posture_risk = 2
            result_text = "⚠️ Обнаружена АСИММЕТРИЯ плеч! Рекомендуется консультация врача."
            # Рисуем красный прямоугольник вокруг плеч
            for shoulder in [left_shoulder, right_shoulder]:
                x = int(shoulder.x * width)
                y = int(shoulder.y * height)
                cv2.rectangle(annotated_image, (x-20, y-20), (x+20, y+20), (0, 0, 255), 3)
        elif left_shoulder.y > 0.6 or right_shoulder.y > 0.6:
            posture_risk = 1
            result_text = "⚠️ Обнаружена СУТУЛОСТЬ! Рекомендуются упражнения для спины."
        else:
            posture_risk = 0
            result_text = "✅ Осанка в норме! Так держать!"
        
        # Добавляем текст на изображение
        cv2.putText(annotated_image, result_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Сохраняем аннотированное изображение
        if output_path is None:
            filename = f"analyzed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            output_path = os.path.join('static', 'uploads', filename)
        
        cv2.imwrite(output_path, annotated_image)
        
        return posture_risk, result_text, output_path