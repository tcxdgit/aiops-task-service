import os
import logging

from app import app
from gunicorn.app.base import BaseApplication
from flask import Flask


class Application(BaseApplication):
    def __init__(self, app, config) -> None:
        self.app = app
        self.config = config
        BaseApplication.__init__(self)

    def load_config(self) -> None:
        for key, value in self.config.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key, value)

    def init(self, parser, opts, args):
        pass

    def load(self) -> Flask:
        return self.app

    def stop(self, *args, **kwargs):
        logging.info("Shutting down AI restAPI service")

gunicorn_app = Application(
    app=app,
    config={
        "bind": "0.0.0.0:8888",
        "workers": os.cpu_count(),
        "timeout": 10000,
        "threads": 5,
    },
)


if __name__ == '__main__':
    gunicorn_app.run()