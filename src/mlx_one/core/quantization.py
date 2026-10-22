"""Lossless MLX checkpoint quantization metadata; no runtime backend imports."""

from collections.abc import Mapping


class UnsupportedQuantization(ValueError):
    code = "UNSUPPORTED_QUANTIZATION"


def quantization_config(value):
    if value is None or value == {}:
        return {}
    if not isinstance(value, Mapping):
        raise UnsupportedQuantization("quantization metadata must be an object")
    if value.get("mode") == "turboquant":
        from mlx_one.quantization.turboquant.config import TQ3Config

        return TQ3Config.from_dict(value).to_dict()
    result = quantization_spec(value)
    for path, override in value.items():
        if path in {"bits", "group_size", "mode", "quant_method", "method"}:
            continue
        if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
            raise UnsupportedQuantization(f"invalid quantization module path: {path!r}")
        if override is False:
            result[path] = False
        elif isinstance(override, Mapping):
            unknown = set(override) - {"bits", "group_size", "mode", "quant_method", "method"}
            if unknown:
                raise UnsupportedQuantization(
                    f"unknown quantization settings for {path}: {unknown}"
                )
            # MLX per-module dictionaries are complete specs, not inherited defaults.
            # In particular GPT-OSS's Q8 routers omit mode and require affine.
            result[path] = quantization_spec(override)
        else:
            raise UnsupportedQuantization(f"invalid quantization override for {path}")
    return result


def quantization_spec(value):
    mode = value.get("mode", "affine")
    bits = value.get("bits")
    group = value.get("group_size", 32 if mode == "mxfp4" else 64)
    method = value.get("quant_method", value.get("method"))
    if method is not None and method not in {"mlx", mode}:
        raise UnsupportedQuantization(f"unsupported quantization method: {method!r}")
    if mode not in {"affine", "mxfp4", "nvfp4"}:
        raise UnsupportedQuantization(f"unsupported MLX quantization mode: {mode!r}")
    allowed = {2, 3, 4, 5, 6, 8} if mode == "affine" else {4}
    groups = {32, 64, 128} if mode == "affine" else {32} if mode == "mxfp4" else {16}
    if type(bits) is not int or bits not in allowed:
        raise UnsupportedQuantization(f"unsupported {mode} quantization bits: {bits!r}")
    if type(group) is not int or group not in groups:
        raise UnsupportedQuantization(f"unsupported {mode} quantization group_size: {group!r}")
    return {"bits": bits, "group_size": group, "mode": mode}


def module_quantization(config, path):
    return config.get(path, {key: config[key] for key in ("bits", "group_size", "mode")})
