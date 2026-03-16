"""trustguard.dataset — PermissionBench loader, builder, and preprocessing."""

from trustguard.dataset.permissionbench_loader import (
    PermissionBenchLoader,
    PermissionBenchDataset,
    parse_permission_vector,
)
from trustguard.dataset.dataset_builder import DatasetBuilder, AppRecord
from trustguard.dataset.preprocessing import (
    normalise_description,
    build_api_feature_string,
    build_api_call_graph,
    annotate_permission_labels,
    stratified_split,
    save_splits,
    TAU_LOW,
    TAU_HIGH,
)

__all__ = [
    "PermissionBenchLoader",
    "PermissionBenchDataset",
    "parse_permission_vector",
    "DatasetBuilder",
    "AppRecord",
    "normalise_description",
    "build_api_feature_string",
    "build_api_call_graph",
    "annotate_permission_labels",
    "stratified_split",
    "save_splits",
    "TAU_LOW",
    "TAU_HIGH",
]
