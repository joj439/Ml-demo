import os
import urllib.request
import sys
import time

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "train.csv")
# Official mirror of the Kaggle / Wikipedia Toxic Comment Classification Challenge dataset
DATA_URL = "https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge/resolve/main/train.csv"

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = min(int(count * block_size * 100 / total_size), 100) if total_size > 0 else 0
    sys.stdout.write(f"\rDownloading dataset: {percent}% ({progress_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB) - {speed} KB/s")
    sys.stdout.flush()

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 1024 * 1024:
        print(f"Dataset already exists at: {DATA_FILE} ({os.path.getsize(DATA_FILE) / (1024*1024):.2f} MB)")
        return DATA_FILE

    print(f"Dataset not found locally. Downloading from public mirror...")
    print(f"Source URL: {DATA_URL}")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)
    
    urllib.request.urlretrieve(DATA_URL, DATA_FILE, reporthook)
    print(f"\nDownload completed successfully: {DATA_FILE}")
    print(f"File size: {os.path.getsize(DATA_FILE) / (1024*1024):.2f} MB")
    return DATA_FILE

if __name__ == "__main__":
    download_dataset()
