"""Central definition of the supported business categories."""

from enum import StrEnum


class Category(StrEnum):
    TECHNICAL = "Технически проблем"
    ACCESS = "Достъп или регистрация"
    CONTENT = "Учебно съдържание"
    OTHER = "Друго"


CATEGORIES = tuple(category.value for category in Category)
LABEL_TO_ID = {label: index for index, label in enumerate(CATEGORIES)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
