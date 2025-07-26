import time
from pathlib import Path
import sys

# Добавляем корень проекта в PYTHONPATH
root_dir = Path(__file__).parent   # папка, где лежит main.py
sys.path.append(str(root_dir))

from src.shot_aggregator import ShotAggregator
from src.utilites.save import save_results_to_json

if __name__ == "__main__":
    start_time = time.time()
    aggregator = ShotAggregator("data/shots_clock_mini")
    probs, detailed_results = aggregator.process()
    save_results_to_json(probs, detailed_results, "/result/json/", "result_test.json")
    print("Probabilities:", probs)
    print("Detailed results:", detailed_results)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Время выполнения: {execution_time:.4f} секунд")