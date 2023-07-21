from abc import ABC

class MetadataBase(ABC):
    """
    Base class that ensures tasks, trainers, and models have the appropriate
    metadata.
    """
    def get_metadata(self):
        return {
            'name': type(self).__name__,
        }
