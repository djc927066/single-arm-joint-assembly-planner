"""Layered lazy step cost for ranking candidate initial poses (see README.md).

hard prune -> joint-time lower bound -> batched straight-line check -> lazy cuRobo refinement
of only the blocked candidates that may still enter the top K.
"""
