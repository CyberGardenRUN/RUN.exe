import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import serial
import threading
import time
from queue import Queue

# Параметры маяков (x, y, z) и их уникальные COM порты
beacons = [
    {'id': 1, 'position': (0, 0, 0), 'port': 'COM8'},   # маяк 1 на порту COM8
    {'id': 2, 'position': (5, 0, 0), 'port': 'COM14'},  # маяк 2 на порту COM14
    {'id': 3, 'position': (0, 5, 0), 'port': 'COM15'},  # маяк 3 на порту COM15
    {'id': 4, 'position': (0, 0, 5), 'port': 'COM16'}   # маяк 4 на порту COM16
]

# Место расположения трекера
tracker_position = (0, 0, 0)  # Изначально на (0, 0, 0)

# Функция для расчета расстояния между двумя точками
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

# Функция для трилатерации
def trilateration(beacons, distances):
    A = np.array([
        [2 * (beacons[1]['position'][0] - beacons[0]['position'][0]), 2 * (beacons[1]['position'][1] - beacons[0]['position'][1]), 2 * (beacons[1]['position'][2] - beacons[0]['position'][2])],
        [2 * (beacons[2]['position'][0] - beacons[0]['position'][0]), 2 * (beacons[2]['position'][1] - beacons[0]['position'][1]), 2 * (beacons[2]['position'][2] - beacons[0]['position'][2])],
        [2 * (beacons[3]['position'][0] - beacons[0]['position'][0]), 2 * (beacons[3]['position'][1] - beacons[0]['position'][1]), 2 * (beacons[3]['position'][2] - beacons[0]['position'][2])]
    ])
    
    B = np.array([
        distances[0]**2 - distances[1]**2 - np.sum(np.array(beacons[0]['position'])**2) + np.sum(np.array(beacons[1]['position'])**2),
        distances[0]**2 - distances[2]**2 - np.sum(np.array(beacons[0]['position'])**2) + np.sum(np.array(beacons[2]['position'])**2),
        distances[0]**2 - distances[3]**2 - np.sum(np.array(beacons[0]['position'])**2) + np.sum(np.array(beacons[3]['position'])**2)
    ])
    
    position = np.linalg.solve(A, B)
    return position

# Функция для извлечения расстояния из строки
def extract_distance(data):
    match = re.search(r"Distance:\s*(-?\d+\.\d+)\s+meters", data)
    if match:
        return float(match.group(1))
    return None

# Функция фильтрации помех (EMA)
def filter_distance(prev_distance, current_distance, alpha=0.3):
    if prev_distance is None:
        return current_distance
    return alpha * current_distance + (1 - alpha) * prev_distance

# Функция для создания 3D графика
def plot_3d(beacons, tracker_position, queue):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Рисуем маяки
    for beacon in beacons:
        ax.scatter(beacon['position'][0], beacon['position'][1], beacon['position'][2], color='r', s=100, label=f"Beacon {beacon['id']}")

    # Рисуем трекер
    ax.scatter(tracker_position[0], tracker_position[1], tracker_position[2], color='b', s=100, label="Tracker")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    while True:
        if not queue.empty():
            tracker_position = queue.get()

        ax.cla()  # Очищаем предыдущие данные
        for beacon in beacons:
            ax.scatter(beacon['position'][0], beacon['position'][1], beacon['position'][2], color='r', s=100, label=f"Beacon {beacon['id']}")
        ax.scatter(tracker_position[0], tracker_position[1], tracker_position[2], color='b', s=100, label="Tracker")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.draw()
        plt.pause(0.1)  # Задержка для обновления графика

# Сетевой сервер для получения данных о расстоянии от маяков
def listen_for_distances(beacons, queue):
    global tracker_position
    serial_connections = []
    message_counters = {beacon['port']: 0 for beacon in beacons}  # Счётчик сообщений для каждого порта
    prev_distances = [None] * len(beacons)  # Предыдущие расстояния для каждого маяка

    for beacon in beacons:
        try:
            ser = serial.Serial(beacon['port'], 115200, timeout=1)
            serial_connections.append(ser)
            print(f"Подключено к порту {beacon['port']} для маяка {beacon['id']}")
        except Exception as e:
            print(f"Ошибка при подключении к порту {beacon['port']} для маяка {beacon['id']}: {e}")
            continue

    while True:
        distances = []
        for i, ser in enumerate(serial_connections):
            port = beacons[i]['port']
            try:
                data = ser.readline().decode('latin-1', errors='ignore').strip()
                if data:
                    message_counters[port] += 1
                    if message_counters[port] > 20:
                        print(f"Данные с порта {ser.portstr}: {data}")
                        distance = extract_distance(data)
                        if distance is not None:
                            distances.append(distance)
                            prev_distances[i] = filter_distance(prev_distances[i], distance)  # Применяем фильтрацию
                        else:
                            print(f"Не удалось извлечь расстояние из данных: {data}")
                            distances.append(0)
                    else:
                        print(f"Пропускаем сообщение с порта {ser.portstr}")
                else:
                    print(f"Пустые данные с порта {ser.portstr}")
                    distances.append(0)
            except Exception as e:
                print(f"Ошибка при получении данных с порта {ser.portstr}: {e}")
                distances.append(0)

        if len(distances) == len(beacons):
            tracker_position = trilateration(beacons, distances)
            print(f"Tracker Position: {tracker_position}")

            queue.put(tracker_position)

        time.sleep(0.1)  # Снижаем нагрузку, делаем паузу

# Основная функция
def main():
    queue = Queue()
    threading.Thread(target=listen_for_distances, args=(beacons, queue), daemon=True).start()
    plot_3d(beacons, tracker_position, queue)

if __name__ == '__main__':
    main()
