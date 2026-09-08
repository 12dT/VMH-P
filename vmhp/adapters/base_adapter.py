from abc import ABC, abstractmethod

from vmhp.core.packet import EvidencePacket


class BaseBackboneAdapter(ABC):
    @abstractmethod
    def extract(self, batch) -> EvidencePacket:
        raise NotImplementedError

    @abstractmethod
    def freeze_backbone(self):
        raise NotImplementedError

    @abstractmethod
    def unfreeze_for_joint_tuning(self):
        raise NotImplementedError
