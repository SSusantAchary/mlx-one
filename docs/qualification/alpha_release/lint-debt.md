# LINT-001: Inherited Ruff debt

Open; release-owner disposition pending. This tracker preserves all 65 entry findings. New findings fail CI. No blanket noqa or XFAIL was added.

Ten findings in CLI, inspection and registry are import ordering/line length. The other 55 are in experimental SAM3 segmentation, outside the proposed text-only envelope. The five closure warnings require a separate SAM3 behavior review; this release does not waive them or advertise SAM3 certification.

| ID | File | Code | Entry line |
|---|---|---|---|
| LINT-001 | `src/mlx_one/cli.py` | E402 | 102 |
| LINT-002 | `src/mlx_one/cli.py` | E501 | 203 |
| LINT-003 | `src/mlx_one/cli.py` | E501 | 204 |
| LINT-004 | `src/mlx_one/cli.py` | E501 | 255 |
| LINT-005 | `src/mlx_one/cli.py` | E501 | 260 |
| LINT-006 | `src/mlx_one/cli.py` | E501 | 287 |
| LINT-007 | `src/mlx_one/inspection.py` | E501 | 673 |
| LINT-008 | `src/mlx_one/models/registry.py` | E501 | 68 |
| LINT-009 | `src/mlx_one/models/registry.py` | E501 | 72 |
| LINT-010 | `src/mlx_one/models/registry.py` | E501 | 73 |
| LINT-011 | `src/mlx_one/models/segmentation/sam3/_configuration.py` | I001 | 3 |
| LINT-012 | `src/mlx_one/models/segmentation/sam3/_runtime.py` | B905 | 306 |
| LINT-013 | `src/mlx_one/models/segmentation/sam3/_runtime.py` | E501 | 554 |
| LINT-014 | `src/mlx_one/models/segmentation/sam3/_runtime.py` | E731 | 1002 |
| LINT-015 | `src/mlx_one/models/segmentation/sam3/_runtime.py` | E731 | 1003 |
| LINT-016 | `src/mlx_one/models/segmentation/sam3/_runtime.py` | E731 | 1004 |
| LINT-017 | `src/mlx_one/models/segmentation/sam3/_runtime.py` | B023 | 1308 |
| LINT-018 | `src/mlx_one/models/segmentation/sam3/_runtime.py` | B023 | 1309 |
| LINT-019 | `src/mlx_one/models/segmentation/sam3/_runtime.py` | B023 | 1309 |
| LINT-020 | `src/mlx_one/models/segmentation/sam3/_runtime.py` | B023 | 1309 |
| LINT-021 | `src/mlx_one/models/segmentation/sam3/_runtime.py` | B023 | 1310 |
| LINT-022 | `src/mlx_one/models/segmentation/sam3/_runtime.py` | B905 | 1318 |
| LINT-023 | `src/mlx_one/models/segmentation/sam3/config.py` | I001 | 6 |
| LINT-024 | `src/mlx_one/models/segmentation/sam3/config.py` | E501 | 659 |
| LINT-025 | `src/mlx_one/models/segmentation/sam3/config.py` | E501 | 668 |
| LINT-026 | `src/mlx_one/models/segmentation/sam3/config.py` | E501 | 681 |
| LINT-027 | `src/mlx_one/models/segmentation/sam3/config.py` | E501 | 685 |
| LINT-028 | `src/mlx_one/models/segmentation/sam3/model.py` | I001 | 6 |
| LINT-029 | `src/mlx_one/models/segmentation/sam3/model.py` | E501 | 244 |
| LINT-030 | `src/mlx_one/models/segmentation/sam3/model.py` | B905 | 995 |
| LINT-031 | `src/mlx_one/models/segmentation/sam3/tracker.py` | I001 | 6 |
| LINT-032 | `src/mlx_one/models/segmentation/sam3/tracker.py` | E501 | 87 |
| LINT-033 | `src/mlx_one/models/segmentation/sam3/tracker.py` | E501 | 308 |
| LINT-034 | `src/mlx_one/models/segmentation/sam3/tracker.py` | E501 | 609 |
| LINT-035 | `src/mlx_one/models/segmentation/sam3/tracker.py` | B905 | 674 |
| LINT-036 | `src/mlx_one/models/segmentation/sam3/tracker.py` | E501 | 715 |
| LINT-037 | `src/mlx_one/models/segmentation/sam3/tracker.py` | B905 | 734 |
| LINT-038 | `src/mlx_one/models/segmentation/sam3/tracker_video.py` | I001 | 6 |
| LINT-039 | `src/mlx_one/models/segmentation/sam3/tracker_video.py` | E501 | 372 |
| LINT-040 | `src/mlx_one/models/segmentation/sam3/tracker_video.py` | B905 | 1329 |
| LINT-041 | `src/mlx_one/models/segmentation/sam3/tracker_video.py` | E501 | 1513 |
| LINT-042 | `src/mlx_one/models/segmentation/sam3/tracker_video.py` | B905 | 1538 |
| LINT-043 | `src/mlx_one/models/segmentation/sam3/tracker_video.py` | UP034 | 1686 |
| LINT-044 | `src/mlx_one/models/segmentation/sam3/tracker_video.py` | E501 | 1703 |
| LINT-045 | `src/mlx_one/models/segmentation/sam3/tracker_video.py` | E501 | 1934 |
| LINT-046 | `src/mlx_one/models/segmentation/sam3/tracker_video.py` | B905 | 1939 |
| LINT-047 | `src/mlx_one/models/segmentation/sam3/tracker_video.py` | E501 | 2059 |
| LINT-048 | `src/mlx_one/models/segmentation/sam3/tracker_video.py` | E501 | 2078 |
| LINT-049 | `src/mlx_one/models/segmentation/sam3/video.py` | I001 | 6 |
| LINT-050 | `src/mlx_one/models/segmentation/sam3/video.py` | E501 | 184 |
| LINT-051 | `src/mlx_one/models/segmentation/sam3/video.py` | B905 | 196 |
| LINT-052 | `src/mlx_one/models/segmentation/sam3/video.py` | B905 | 197 |
| LINT-053 | `src/mlx_one/models/segmentation/sam3/video.py` | B905 | 198 |
| LINT-054 | `src/mlx_one/models/segmentation/sam3/video.py` | E501 | 606 |
| LINT-055 | `src/mlx_one/models/segmentation/sam3/video.py` | E501 | 642 |
| LINT-056 | `src/mlx_one/models/segmentation/sam3/video.py` | E501 | 936 |
| LINT-057 | `src/mlx_one/models/segmentation/sam3/video.py` | B905 | 949 |
| LINT-058 | `src/mlx_one/models/segmentation/sam3/video.py` | B905 | 1029 |
| LINT-059 | `src/mlx_one/models/segmentation/sam3/video.py` | B905 | 1031 |
| LINT-060 | `src/mlx_one/models/segmentation/sam3/video.py` | B905 | 1051 |
| LINT-061 | `src/mlx_one/models/segmentation/sam3/video.py` | B905 | 1098 |
| LINT-062 | `src/mlx_one/models/segmentation/sam3/video.py` | B905 | 1111 |
| LINT-063 | `src/mlx_one/models/segmentation/sam3/video.py` | B905 | 1210 |
| LINT-064 | `src/mlx_one/models/segmentation/sam3/video.py` | E501 | 1285 |
| LINT-065 | `src/mlx_one/models/segmentation/sam3/video.py` | E501 | 1409 |
