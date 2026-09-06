# Security Policy

## Supported versions

Until 1.0, only the most recent published release receives security fixes.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
security advisory flow for the `SSusantAchary/mlx-one` repository. Include the
affected version, reproduction steps, impact, and any suggested mitigation.

The maintainer will acknowledge a report within seven days and will coordinate a
fix and disclosure timeline after validation.

## Model and dataset safety

Model repositories and datasets are external inputs. Review their code, licenses,
and trust requirements before use. `mlx-one` will not enable remote model code by
default. Generated code execution and arbitrary pipeline shell steps are outside
the v1 scope.
