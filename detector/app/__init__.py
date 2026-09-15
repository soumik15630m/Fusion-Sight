"""Detector service package.

Fusion used to be an in-process import here (repo_root/fusion on sys.path); it
is now a separate service reached over the network (app/fusion_client.py), so
this package no longer touches the sibling `fusion/` tree.
"""
