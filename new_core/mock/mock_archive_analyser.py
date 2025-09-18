

from new_core.interfaces.archive_analyser import IArchiveAnalyser
from new_core.interfaces.archive_store import IArchiveStore
from new_core.models.archive_feedback import ArchiveFeedback
from new_core.models.creative_strategy import CreativeStrategy
from new_core.models.run_config import RunConfig


class MockArchiveAnalyser(IArchiveAnalyser):
    async def generate_feedback(self, run_config: RunConfig, archive: IArchiveStore, current_strategy: CreativeStrategy) -> ArchiveFeedback:
        return ArchiveFeedback(
            text="No feedback. No change needed to current strategy."
        )

    