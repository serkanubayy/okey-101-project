An advanced computer vision project designed to detect, track, and analyze Okey-101 tiles in real-time using YOLOv8 and Custom CNN models.



## 🚀 Features
- **Real-time Tile Detection:** Powered by YOLOv8 for high-speed bounding box detection.
- **Custom CNN Classification:** Specialized Convolutional Neural Networks for high-accuracy number and color recognition.
- **Temporal Tracking:** Eliminates flickering and provides stable ID assignment across frames.
- **Smart Game Logic:** Analyzes the best possible "Per" (melds) and "Pair" combinations according to Okey-101 rules.
- **Ghost Buster Mode:** Automatically handles unreadable tiles by assigning them as "Jokers" based on game context.

## 🛠️ Tech Stack
- **Python 3.10+**
- **OpenCV:** Image preprocessing and camera stream management.
- **PyTorch:** Custom CNN model inference.
- **Ultralytics (YOLOv8):** Object detection.
- **NumPy:** Mathematical operations and spatial cleanup.

## 📂 Project Structure
- `main_v2.py`: The core execution script (Golden Standard).
- `okey_zeka.py`: The "AI Brain" that calculates game scores and melds.
- `cnn/`: Contains logic for number and color classification models.
- `temporal_tracker.py`: Ensures ID stability between frames.
- `tile_id_manager.py`: Manages unique IDs for each detected tile.

## ⚙️ Installation & Usage
1. Install dependencies:
   ```bash
   pip install opencv-python ultralytics torch torchvision pillow tqdm