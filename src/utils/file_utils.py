import os
import pickle
import json
import joblib
from pathlib import Path
from typing import Any, Dict, List, Union
import logging

logger = logging.getLogger(__name__)


def ensure_dir(directory: Union[str, Path]) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    # Chuyển directory thành đối tượng Path
    # parents=True: Tạo các thư mục cha nếu chưa tồn tại
    # exist_ok=True: Không ném lỗi nếu thư mục đã tồn tại


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    # Chuyển filepath thành đối tượng Path
    filepath = Path(filepath)
    # ensure_dir tạo thư mục cha của filepath nếu null
    ensure_dir(filepath.parent)

    with open(filepath, 'wb') as f: # wb: write binary
        pickle.dump(obj, f) # save obj
    logger.info(f"Saved pickle file: {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Tải một đối tượng Python từ tệp pickle"""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'rb') as f: # read binary
        obj = pickle.load(f) # # save obj
    logger.info(f"Loaded pickle file: {filepath}")
    return obj
    # Lưu các đối tượng Python phức tạp


def save_json(obj: Dict, filepath: Union[str, Path], indent: int = 2) -> None: # mức thụt lề
    """Lưu một từ điển Python vào tệp JSON"""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, 'w', encoding='utf-8') as f: # write
        json.dump(obj, f, indent=indent, ensure_ascii=False) # thụt 2, unicode
    logger.info(f"Saved JSON file: {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict:
    """Tải một từ điển từ tệp JSON"""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f: # read
        obj = json.load(f) # load dic
    logger.info(f"Loaded JSON file: {filepath}")
    return obj


def save_model(model: Any, filepath: Union[str, Path]) -> None:
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    joblib.dump(model, filepath) # save model
    logger.info(f"Saved model: {filepath}")


def load_model(filepath: Union[str, Path]) -> Any:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    model = joblib.load(filepath) # load model
    logger.info(f"Loaded model: {filepath}")
    return model


def get_file_size(filepath: Union[str, Path]) -> str:
    filepath = Path(filepath)
    if not filepath.exists():
        return "File not found"

    size = filepath.stat().st_size

    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def list_files(directory: Union[str, Path], pattern: str = "*") -> List[Path]:
    # pattern: mẫu (all), dic: dir/Path
    """Tìm tệp"""
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    files = list(directory.glob(pattern))
    logger.info(f"Found {len(files)} files matching '{pattern}' in {directory}")
    return files


def clean_directory(directory: Union[str, Path], keep_files: List[str] = None) -> None:
    """Clean directory, optionally keeping specific files"""
    directory = Path(directory)
    if not directory.exists():
        return

    keep_files = keep_files or [] # file or null

    for file_path in directory.iterdir():
        if file_path.name not in keep_files: # k thuộc keep_file
            if file_path.is_file(): # tệp
                file_path.unlink()
            elif file_path.is_dir(): # folder
                clean_directory(file_path)
                file_path.rmdir()

    logger.info(f"Cleaned directory: {directory}")


def backup_file(filepath: Union[str, Path], backup_dir: Union[str, Path] = None) -> Path:
    """Tạo bản sao lưu"""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if backup_dir is None:
        backup_dir = filepath.parent / "backup"
    else:
        backup_dir = Path(backup_dir)

    ensure_dir(backup_dir)

    import shutil
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
    backup_path = backup_dir / backup_name

    shutil.copy2(filepath, backup_path) # copy tệp
    logger.info(f"Created backup: {backup_path}")

    return backup_path