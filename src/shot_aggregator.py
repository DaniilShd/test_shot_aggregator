from ultralytics import YOLO
import torch
from config.config_loader import load_config
from config.scheme import ProcessingVideoConfig, TrackerConfig, AppConfig, SimilarityConfig, CompareThreshold
import cv2
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity
import contextlib
import io
import logging


class ShotAggregator:
    """
    Класс для обработки видео, детекции людей, трекинга и сравнения эмбеддингов.

    Attributes:
        path_files (str): Путь к директории с видеофайлами.
        model (YOLO): Модель YOLO для детекции объектов.
        skip_frames (int): Количество пропускаемых кадров.
        sharp_threshold (float): Порог резкости для фильтрации кадров.
        confidence (float): Порог уверенности для детекции.
    """

    def __init__(self, path_files, model_path='yolov8n-seg.pt'):
        """
        Инициализация ShotAggregator.

        Args:
            path_files (str): Путь к директории с видеофайлами.
            model_path (str): Путь к файлу модели YOLO.
        """
        self.path_files = path_files
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Подавление вывода YOLO при загрузке модели
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = YOLO(model_path).to(device)

        # Загрузка конфигурации
        try:
            config = load_config()
        except Exception as e:
            logging.warning(f"Ошибка загрузки конфига: {e}. Используются значения по умолчанию.")
            config = AppConfig(
                processing=ProcessingVideoConfig(),
                tracker=TrackerConfig(),
                similarity=SimilarityConfig(),
                compare=CompareThreshold(),
            )

        self.skip_frames = config.processing.skip_frames
        self.sharp_threshold = config.processing.sharp_threshold
        self.confidence = config.processing.confidence
        self.max_age = config.tracker.max_age
        self.n_init = config.tracker.n_init
        self.nn_budget = config.tracker.nn_budget
        self.similarity = config.similarity.alpha
        self.compare_threshold = config.compare.threshold

    def calculate_sharpness(self, image):
        """
        Вычисляет меру резкости изображения с помощью оператора Лапласа.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            float: Мера резкости.
        """
        if image.size == 0:
            return 0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    def process_video(self, file_path):
        """
        Обрабатывает видеофайл: детектирует людей, трекает их и сохраняет эмбеддинги.

        Args:
            file_path (str): Путь к видеофайлу.

        Returns:
            defaultdict: Словарь с эмбеддингами для каждого трека. self.max_age, self.n_init, self.nn_budget
        """
        cap = cv2.VideoCapture(file_path)
        tracker = DeepSort(max_age=self.max_age, n_init=self.n_init, nn_budget=self.nn_budget)
        saved_embeddings = defaultdict(list)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_count in tqdm(range(total_frames), desc=f"Обработка {os.path.basename(file_path)}"):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (self.skip_frames + 1) != 0:
                continue

            # Подготовка размытого фона
            blurred_bg = cv2.GaussianBlur(frame, (51, 51), 0)
            masked_frame = blurred_bg.copy()

            # Подавление вывода YOLO при детекции
            results = self.model(frame, classes=[0, 6, 7, 8, 9, 10])

            if results is not None:
                for result in results:
                    if result.masks is not None:
                        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        for mask in result.masks.data:
                            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
                            mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                            combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

                        inverse_mask = cv2.bitwise_not(combined_mask)
                        foreground = cv2.bitwise_and(frame, frame, mask=combined_mask)
                        background = cv2.bitwise_and(blurred_bg, blurred_bg, mask=inverse_mask)
                        masked_frame = cv2.add(foreground, background)

                    detections = []
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        roi = frame[y1:y2, x1:x2]

                        if roi.size == 0:
                            continue

                        if (self.calculate_sharpness(roi) > self.sharp_threshold or
                                (x2 - x1) * (y2 - y1) > (frame.shape[0] * frame.shape[1]) / 8):
                            conf = float(box.conf[0].cpu().numpy())
                            if conf > self.confidence:
                                detections.append([[x1, y1, x2 - x1, y2 - y1], conf, None])

                    tracks = tracker.update_tracks(detections, frame=masked_frame)

                    for track in tracks:
                        if not track.is_confirmed():
                            continue

                        track_id = track.track_id
                        embedding = track.features

                        if embedding is not None and track_id not in saved_embeddings:
                            saved_embeddings[track_id].append(embedding)

        cap.release()

        return saved_embeddings

    def compare_adjacent_videos(self, video_embeddings_list):
        """
        Сравнивает эмбеддинги между соседними видео.

        Args:
            video_embeddings_list (list): Список эмбеддингов для каждого видео.
            threshold (float): Порог сходства.

        Returns:
            list: Результаты сравнения для каждой пары видео.
        """
        compare = []

        for i in range(len(video_embeddings_list) - 1):
            if not video_embeddings_list[i] or not video_embeddings_list[i + 1]:
                compare.append({
                    'video_pair': (i, i + 1),
                    'pairwise_similarity': None,
                    'match_rate': None,
                    'error': 'Empty embedding list'
                })
                continue

            try:
                emb1_list = [np.array(e).reshape(1, -1) for e in video_embeddings_list[i]]
                emb2_list = [np.array(e).reshape(1, -1) for e in video_embeddings_list[i + 1]]

                emb_len = emb1_list[0].shape[1]
                if any(e.shape[1] != emb_len for e in emb1_list + emb2_list):
                    raise ValueError("Embeddings have different dimensions")

                sim_matrix = cosine_similarity(np.vstack(emb1_list), np.vstack(emb2_list))

                compare.append({
                    'video_pair': (i, i + 1),
                    'pairwise_similarity': float(np.mean(sim_matrix)),
                    'match_rate': float(100 * np.mean(sim_matrix > self.compare_threshold))
                })

            except Exception as e:
                compare.append({
                    'video_pair': (i, i + 1),
                    'pairwise_similarity': None,
                    'match_rate': None,
                    'error': str(e)
                })

        # Параметры
        alpha = self.similarity  # Вес для similarity

        probs = []
        for pair in compare:
            sim = pair['pairwise_similarity']
            match = pair['match_rate']

            if sim is None or match is None:
                # Если данные отсутствуют, считаем вероятность 0
                probs.append(0.5)
                continue

            # Нормализуем match_rate (делим на 100)
            match_norm = match / 100.0

            # Линейная комбинация similarity и match_rate
            p = alpha * sim + (1 - alpha) * match_norm
            probs.append(p)

        # Преобразуем в numpy array
        probs = np.array(probs)

        return probs

    @staticmethod
    def convert_arrays_to_lists(obj):
        """
        Рекурсивно преобразует numpy массивы в списки.

        Args:
            obj: Объект для преобразования.

        Returns:
            Объект с преобразованными массивами.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [ShotAggregator.convert_arrays_to_lists(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: ShotAggregator.convert_arrays_to_lists(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def dict_to_list(data):
        """
        Преобразует словарь эмбеддингов в список.

        Args:
            data (dict): Словарь эмбеддингов.

        Returns:
            list: Список эмбеддингов.
        """
        return [item[0][0] for _, item in data.items()] if data else []

    def process(self):
        """
        Основной метод обработки всех видео в директории.

        Returns:
            list: Результаты сравнения соседних видео.
        """
        files = sorted([f for f in os.listdir(self.path_files) if f.endswith(".mp4")])
        embeddings = []

        for file in files:
            saved_embeddings = self.process_video(os.path.join(self.path_files, file))
            data_dict = dict(saved_embeddings)
            data = self.convert_arrays_to_lists(data_dict)
            embeddings.append(self.dict_to_list(data))

        return self.compare_adjacent_videos(embeddings)


if __name__ == "__main__":
    aggregator = ShotAggregator("../data/shots_output_catch_me")
    results = aggregator.process()
    print(results)