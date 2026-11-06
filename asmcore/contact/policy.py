"""Explicit assembly contact decisions on top of immutable geometric evidence."""
from dataclasses import dataclass, replace
import numpy as np
from ..model import digest


@dataclass(frozen=True)
class ToleranceContactPolicy:
    """Treat extracted proximity regions as assembly contacts within a tolerance.

    ``allow_unsigned`` explicitly permits unsigned near bands as surface
    proximity contacts, without certifying inside/outside or physical support.
    The policy never changes geometric classification, active area or overlap.
    Extract with ``ContactConfig(near_tol_m=contact_tol_m)`` to obtain the whole
    requested band. Applying a smaller tolerance to an existing wider patch
    returns undecided; it cannot clip geometry from centroid samples alone.
    """
    contact_tol_m: float
    allow_unsigned: bool = False

    def __post_init__(self):
        if not np.isfinite(self.contact_tol_m) or self.contact_tol_m <= 0:
            raise ValueError('contact_tol_m must be finite and positive')
        if not isinstance(self.allow_unsigned, bool):
            raise TypeError('allow_unsigned must be bool')

    def apply(self, analysis):
        """Annotate patches and pair summaries; return a new ContactAnalysis.

        ``is_contact`` is True, False or None (undecided). Areas refer to
        accepted extracted regions on each side, never their sum or a force
        bearing area. An empty/rejected candidate set is not a free-space proof.
        """
        patches = []
        for patch in analysis.patches:
            decision, reason = None, 'insufficient_distance_evidence'
            unsigned = patch.provenance.get('distance_interval') == 'unsigned'
            lo, hi = patch.gap_interval_m
            band = patch.provenance.get('band_limit_m')
            within = (max(abs(lo), abs(hi)) <= self.contact_tol_m or
                      (band is not None and 0 < band <= self.contact_tol_m))
            if patch.classification == 'interference':
                decision, reason = False, 'penetration_is_not_tolerance_contact'
            elif patch.classification == 'active':
                decision, reason = True, 'existing_active_region'
            elif unsigned and not self.allow_unsigned:
                reason = 'unsigned_distance_not_enabled'
            elif lo > self.contact_tol_m:
                decision, reason = False, 'outside_contact_tolerance'
            elif not within:
                reason = 'reextract_band_at_contact_tolerance'
            elif unsigned and patch.measure_kind == 'near_band':
                decision, reason = True, 'unsigned_surface_proximity_assumption'
            elif patch.classification == 'near':
                decision, reason = True, 'positive_gap_within_contact_tolerance'
            # A signed interval crossing zero may include penetration. It is
            # not accepted merely because its magnitude is small.
            record = {'is_contact': decision, 'reason': reason,
                      'contact_tol_m': self.contact_tol_m, 'allow_unsigned': self.allow_unsigned,
                      'geometry_classification': patch.classification,
                      'quality': patch.quality, 'physical_contact_area': False}
            patches.append(replace(patch, provenance={**patch.provenance, 'contact_policy': record}))
        diagnostics = []
        for diag in analysis.pair_diagnostics:
            pair = [p for p in patches if (p.part_a, p.part_b) == (diag['part_a'], diag['part_b'])]
            summary = {'contact_tol_m': self.contact_tol_m, 'allow_unsigned': self.allow_unsigned,
                       'scope': 'extracted_regions_only', 'physical_contact_area': False,
                       'sides': {}}
            for side in ('a', 'b'):
                found = [p for p in pair if p.sampling_side == side]
                accepted = [p for p in found if p.provenance['contact_policy']['is_contact'] is True]
                undecided = [p for p in found if p.provenance['contact_policy']['is_contact'] is None]
                summary['sides'][side] = {
                    'status': 'contact' if accepted else 'unknown' if undecided else 'no_accepted_regions',
                    'accepted_area_m2': sum(p.area_m2 for p in accepted),
                    'undecided_area_m2': sum(p.area_m2 for p in undecided)}
            diagnostics.append({**diag, 'contact_policy': summary})
        return replace(analysis, patches=tuple(patches), pair_diagnostics=tuple(diagnostics),
                       state_digest=digest((analysis.state_digest, 'tolerance_contact_policy/1', self)))
