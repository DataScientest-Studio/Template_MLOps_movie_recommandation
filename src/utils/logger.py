import logging


logger = logging.getLogger("Reco_film")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
