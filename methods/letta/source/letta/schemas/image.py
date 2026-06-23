from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from letta.log import get_logger
from letta.schemas.letta_base import LettaBase

logger = get_logger(__name__)

class BaseImage(LettaBase):
    __id_prefix__ = "image"

class Image(BaseImage):
    """
    Representation of an image.

    Parameters:
        id (str): The unique identifier of the tool.
        image_data (str): The base64 encoding of the image data.
    """

    id: str = BaseImage.generate_id_field()
    organization_id: Optional[str] = Field(None, description="The unique identifier of the organization associated with the document.")
    image_data: Optional[str] = Field(None, description="The base64 encoding of the image data.")

class ImageCreate(LettaBase):
    image_data: Optional[str] = Field(None, description="The base64 encoding of the image data.")

class ImageUpdate(LettaBase):
    image_data: Optional[str] = Field(None, description="The base64 encoding of the image data.")
