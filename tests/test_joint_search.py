"""Cost accounting and coupled order/layout search against exhaustive references."""
from itertools import permutations, product
import numpy as np
import pytest
from jbls.search import Placement, joint_search, cost_breakdown


def place(key, x):
    T = np.eye(4); T[0, 3] = x
    return Placement(key, T)


def test_joint_matches_bruteforce_and_pays_unknown_predecessor_once():
    parts = ('a', 'b', 'c')
    goals = {p: np.array([x, 0., 0.]) for p, x in zip(parts, (1., 4., 8.))}
    home = np.array([-1., 0., 0.])
    domains = {p: (place('left', x), place('right', x+2))
               for p, x in zip(parts, (0., 3., 6.))}
    # Synthetic sequence-dependent obstruction: b's left placement cannot wait
    # during c. This tests using suffix placements, not just independent minima.
    def check(node, remaining, pid, x, structure):
        if pid == 'c' and 'b' in node.layout and node.layout['b'].key == 'left':
            return None, 'blocked'
        return {}, ''
    nodes, _ = joint_search(parts, domains, goals, home, lambda r,p: ({}, ''), check, beam_width=1000)
    expected = []
    for order in permutations(parts):
        for picks in product(*(domains[p] for p in parts)):
            layout = dict(zip(parts, picks))
            if order.index('c') < order.index('b') and layout['b'].key == 'left':
                continue
            expected.append(sum(r['total_m'] for r in cost_breakdown(order, layout, goals, home)))
    assert len(nodes) == len(expected)
    assert nodes[0].cost == pytest.approx(min(expected))
    for n in nodes:
        assert n.cost == pytest.approx(sum(r['total_m'] for r in cost_breakdown(n.order, n.layout, goals, home)))


def test_single_part_home_and_no_return():
    nodes, _ = joint_search(('a',), {'a': (place('x', 2),)}, {'a': np.array([5., 0, 0])},
                            np.zeros(3), lambda r,p: ({}, ''), lambda *args: ({}, ''))
    assert nodes[0].cost == 5


def test_structural_failure_never_queries_layout_and_fixed_order_is_enforced():
    domains = {p: (place(p, 1),) for p in ('a', 'b')}
    goals = {p: np.zeros(3) for p in domains}
    checked = []
    def structure(r, p):
        assert isinstance(r, frozenset)  # adapter caches structural subsets
        return (None, 'unstable') if p == 'a' and len(r) == 2 else ({}, '')
    def layout(node, r, p, x, s):
        checked.append((len(r), p)); return {}, ''
    nodes, _ = joint_search(tuple(domains), domains, goals, np.zeros(3), structure, layout)
    assert nodes[0].order == ('a', 'b')
    assert (2, 'a') not in checked
    nodes, report = joint_search(tuple(domains), domains, goals, np.zeros(3), structure,
                                layout, fixed_order=('b', 'a'))
    assert nodes == [] and report['status'] == 'no_candidate_in_budget'


def test_empty_domain_reports_no_candidate_without_calling_oracles():
    def unexpected(*args):
        raise AssertionError('oracle called for empty domain')
    nodes, report = joint_search(('a',), {'a': []}, {'a': np.zeros(3)},
                                 np.zeros(3), unexpected, unexpected)
    assert not nodes and report['status'] == 'empty_domain'
    assert report['wall_s'] == 0 and not report['full_motion_verified']
