"""Schemas module."""

from pydantic import BaseModel


class ImageData(BaseModel):
    """ImageData model."""

    image: str
