from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

# TODO everything in functions should live in this model
from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.image import Image as PydanticImage

if TYPE_CHECKING:
    from letta.orm.organization import Organization


class Image(SqlalchemyBase, OrganizationMixin):
    """Represents an available tool that the LLM can invoke.

    NOTE: polymorphic inheritance makes more sense here as a TODO. We want a superset of images
    that are always available, and a subset scoped to the organization. Alternatively, we could use the apply_access_predicate to build
    more granular permissions.
    """

    __tablename__ = "images"
    __pydantic_model__ = PydanticImage

    image_data: Mapped[str] = mapped_column(doc="The base64 encoding of the image data.")

    # relationships
    organization: Mapped["Organization"] = relationship("Organization", back_populates="images", lazy="selectin")