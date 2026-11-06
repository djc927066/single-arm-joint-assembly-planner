"""Headless assembly geometry in metres, without robot or viewer startup."""
from .model import (Assembly, AssemblyState, ContactAnalysis, ContactConfig,
                    ContactPatch, GeometryConfig, MatingRelation, MeshData,
                    Part, PreparedMesh, Region, SurfacePatch)
from .io import load_assembly, save_assembly, save_report

__all__ = ['Assembly', 'AssemblyState', 'ContactAnalysis', 'ContactConfig',
           'ContactPatch', 'GeometryConfig', 'MatingRelation', 'MeshData',
           'Part', 'PreparedMesh', 'Region', 'SurfacePatch', 'load_assembly',
           'save_assembly', 'save_report', 'analyze_pair', 'analyze_contacts',
           'ContactModel', 'ContactAnalyzer', 'ContactBackend', 'MeshContactBackend',
           'SDFContactBackend', 'SDFConfig', 'GridSDF', 'SignedDistanceField', 'SDFSamples', 'SDFCollisionChecker',
           'ToleranceContactPolicy', 'ContactGraph', 'build_contact_graph', 'ConstraintConfig',
           'LocalConstraints', 'MotionCandidates', 'contact_constraints', 'candidate_motions', 'rebase_twist',
           'DirectionConfig', 'DirectionResult', 'fibonacci_directions', 'solve_directions', 'assembly_directions',
           'ExternalWrench', 'LoadCase', 'SupportCandidate', 'StabilityConfig', 'EquilibriumCase',
           'EquilibriumResult', 'SupportSearchResult', 'check_equilibrium', 'find_support_requirements',
           'StabilitySweepConfig', 'DirectionalLimit', 'StabilitySweepResult', 'DirectionalStabilityAnalyzer',
           'disturbance_directions', 'analyze_directional_stability', 'limit_wrench',
           'MotionConfig', 'RemovalAction', 'ContactPolicy', 'PathValidation', 'RemovalResult', 'validate_object_path', 'plan_removal', 'sample_path',
           'HandlingCapability', 'AuxiliarySupport', 'SequenceConfig', 'SequenceStep', 'SequenceResult', 'SequenceEvaluator', 'plan_sequence', 'plan_assembly', 'replay_sequence',
                  'AssemblabilityScore', 'StepQuality', 'score_assemblability', 'sequence_quality', 'quality_upper_bound',
           'check_force_closure',
           'QualityTransition', 'QualitySearchConfig', 'QualitySequenceResult', 'quality_depth_first', 'plan_quality_sequence',
           ]


def __getattr__(name):
    if name in ('AssemblabilityScore', 'StepQuality', 'score_assemblability', 'sequence_quality', 'quality_upper_bound'):
        from .planning import quality
        return getattr(quality,name)
    if name == 'check_force_closure':
        from .mechanics.force_closure import check_force_closure
        return check_force_closure
    if name in ('QualityTransition', 'QualitySearchConfig', 'QualitySequenceResult', 'quality_depth_first', 'plan_quality_sequence'):
        from .planning import quality_search
        return getattr(quality_search,name)
    if name in ('StabilitySweepConfig', 'DirectionalLimit', 'StabilitySweepResult', 'DirectionalStabilityAnalyzer',
                'disturbance_directions', 'analyze_directional_stability', 'limit_wrench'):
        from .mechanics import stability_sweep
        return getattr(stability_sweep,name)
    if name in ('MotionConfig', 'RemovalAction', 'ContactPolicy', 'PathValidation', 'RemovalResult', 'validate_object_path', 'plan_removal', 'sample_path'):
        from .motion import part_motion
        return getattr(part_motion,name)
    if name in ('HandlingCapability', 'AuxiliarySupport', 'SequenceConfig', 'SequenceStep', 'SequenceResult', 'SequenceEvaluator', 'plan_sequence', 'plan_assembly', 'replay_sequence'):
        from .planning import sequence
        return getattr(sequence,name)
    if name in ('ExternalWrench', 'LoadCase', 'SupportCandidate', 'StabilityConfig', 'EquilibriumCase',
                'EquilibriumResult', 'SupportSearchResult', 'check_equilibrium', 'find_support_requirements'):
        from .mechanics import equilibrium as stability
        return getattr(stability, name)
    if name in ('DirectionConfig', 'DirectionResult', 'fibonacci_directions', 'solve_directions', 'assembly_directions'):
        from .motion import directions
        return getattr(directions, name)
    if name in ('ContactGraph', 'build_contact_graph'):
        from .contact import graph
        return getattr(graph, name)
    if name in ('ConstraintConfig', 'LocalConstraints', 'MotionCandidates',
                'contact_constraints', 'candidate_motions', 'rebase_twist'):
        from .motion import constraints
        return getattr(constraints, name)
    if name == 'ToleranceContactPolicy':
        from .contact.policy import ToleranceContactPolicy
        return ToleranceContactPolicy
    if name == 'SDFCollisionChecker':
        from .contact.sdf_collision import SDFCollisionChecker
        return SDFCollisionChecker
    if name == 'ContactModel':
        from .contact.models import ContactModel
        return ContactModel
    if name in ('ContactAnalyzer', 'ContactBackend', 'MeshContactBackend'):
        from .contact import backends
        return getattr(backends, name)
    if name in ('SDFContactBackend', 'SDFConfig'):
        from .contact import sdf_backend
        return getattr(sdf_backend, name)
    if name in ('GridSDF', 'SignedDistanceField', 'SDFSamples'):
        from .geometry import sdf
        return getattr(sdf, name)
    if name in ('analyze_pair', 'analyze_contacts'):
        from .contact import analysis
        return getattr(analysis, name)
    raise AttributeError(name)
