import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Get BTL directory
    btl_dir = Path(__file__).parent

    # Add BTL to Python path
    if str(btl_dir) not in sys.path:
        sys.path.insert(0, str(btl_dir))

    # Create necessary directories
    dirs_to_create = [
        btl_dir / "saved_models",
        btl_dir / "src" / "web" / "templates"
    ]

    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)

    try:
        # Import and run the app
        from src.web.app import movie_app

        logger.info("Starting Movie Recommendation System...")
        logger.info("Access at: http://127.0.0.1:5000")

        # Run the app
        movie_app.run(host='127.0.0.1', port=5000, debug=True)

    except ImportError as e:
        logger.error(f"Failed to import app: {e}")
        logger.error("Install dependencies: pip install -r requirements.txt")
        return 1
    except Exception as e:
        logger.error(f"Failed to run app: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())