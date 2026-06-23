import uuid
import importlib
import warnings
from typing import List, Optional

from letta.constants import (
    BASE_FUNCTION_RETURN_CHAR_LIMIT,
    BASE_MEMORY_TOOLS,
    BASE_SLEEPTIME_TOOLS,
    BASE_TOOLS,
    LETTA_TOOL_SET,
    MCP_TOOL_TAG_NAME_PREFIX,
    MULTI_AGENT_TOOLS,
)
from letta.functions.functions import derive_openai_json_schema, load_function_set
from letta.log import get_logger

# TODO: Remove this once we translate all of these to the ORM
from letta.orm.errors import NoResultFound
from letta.orm.image import Image as ImageModel
from letta.schemas.image import Image as PydanticImage
from letta.schemas.image import ImageCreate, ImageUpdate
from letta.schemas.user import User as PydanticUser
from letta.utils import enforce_types, printd

logger = get_logger(__name__)

class ImageManager:
    """Manager class to handle image-related operations."""

    def __init__(self):
        # Fetching the db_context similarly as in OrganizationManager
        from letta.server.db import db_context

        self.session_maker = db_context

    def get_image_by_id(self, image_id: str) -> Optional[PydanticImage]:
        """Fetch an image by its ID."""
        with self.session_maker() as session:
        
            try:
                image = ImageModel.read(db_session=session, identifier=image_id)
                return image.to_pydantic()
            except NoResultFound:
                raise NoResultFound(f"Image with id {image_id} not found.")

    @enforce_types
    def create_image(self, pydantic_image: PydanticImage, actor: PydanticUser) -> ImageModel:
        """Create a new image in the database."""

        image_dict = pydantic_image.model_dump()

        required_fields = ['image_data']
        for field in required_fields:
            if field not in image_dict or image_dict[field] is None:
                raise ValueError(f"Required field '{field}' is missing or None in image data")

        # Set defaults if needed
        image_dict.setdefault("id", str(uuid.uuid4()))

        with self.session_maker() as session:
            image = ImageModel(**pydantic_image.dict())
            image.create(session)
            return image.to_pydantic()

    @enforce_types
    def insert_image(
        self,
        image_data: str,
        actor: PydanticUser
    ) -> List[PydanticImage]:
        """Insert a new image into the database."""

        image = self.create_image(
            PydanticImage(
                organization_id=actor.organization_id,
                image_data=image_data
            ),
            actor=actor
        )
        return image