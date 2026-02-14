from typing import Literal

ReporterType = Literal["individual", "ngo", "government"]

TRUST_SCORES = {
    "individual": 0.4,
    "ngo": 0.75,
    "government": 0.9,
}


class TrustService:

    def get_trust_score(self, reporter_type: ReporterType) -> float:
        return TRUST_SCORES.get(reporter_type, 0.3)

    def compute_effective_score(
        self,
        risk_score: float,
        reporter_type: ReporterType
    ) -> float:
        trust = self.get_trust_score(reporter_type)
        return risk_score * trust


_trust = TrustService()


def get_trust_service() -> TrustService:
    return _trust