import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r'config\config.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file in the root directory
env_path = Path(__file__).parent.parent / '.env'
if not env_path.exists():
    logger.error(f"Environment file not found at {env_path}. Please ensure it exists.")
    raise FileNotFoundError(f".env file not found at {env_path}")

load_dotenv(env_path)

# Required environment variables
REQUIRED_VARS = ['ASSEMBLYAI_API_KEY', 'ELEVENLABS_API_KEY', 'VOICE_ID']

# Load and validate environment variables
config = {}
missing_vars = []

for var in REQUIRED_VARS:
    value = os.getenv(var)
    if not value:
        missing_vars.append(var)
    config[var] = value

if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise ValueError(error_msg)

# Export variables
ASSEMBLYAI_API_KEY = config['ASSEMBLYAI_API_KEY']
ELEVENLABS_API_KEY = config['ELEVENLABS_API_KEY']
VOICE_ID = config['VOICE_ID']

# ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# VOICE_ID = os.getenv("VOICE_ID")

logger.info("Configuration loaded successfully")
logger.info("Loaded configuration values:")
for key, value in config.items():
    # Mask sensitive data for logging
    display_value = value if len(value) <= 4 else f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"
    logger.info(f"{key}: {display_value}")