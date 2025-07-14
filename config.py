import os


class ModalSettings:
    def __init__(self):
        self.inference_app_id = os.environ.get(
            "MODAL_INFERENCE_APP_ID", "outpaint-inference-dev"
        )
        self.web_app_id = os.environ.get("MODAL_WEB_APP_ID", "outpaint-web-dev")
        self.max_batch_size = int(os.environ.get("MODAL_BATCH_SIZE", 50))
        self.wait_ms = int(os.environ.get("MODAL_WAIT_MS", 5000))
        self.gpu = os.environ.get("MODAL_GPU", "A10G")
        self.timeout = int(os.environ.get("MODAL_TIMEOUT_MINS", 30)) * 60  # in seconds


# TODO: Add more settings here
class Settings:
    def __init__(self):
        self.api_key = os.environ.get("OUTPAINT_API_KEY")


settings = Settings()
modal_settings = ModalSettings()
