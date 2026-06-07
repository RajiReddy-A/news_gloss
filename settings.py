"""Load local environment variables without overriding deployment settings."""

from dotenv import load_dotenv


def load_environment() -> None:
    load_dotenv(override=False)

