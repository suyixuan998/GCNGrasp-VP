from __future__ import annotations

from types import SimpleNamespace

from .path_extension import Path


def parse_seq(value: str) -> list[int]:
    """ Parse a sequence string like "1-3,5,7-8" to a list of integers [1,2,3,5,7,8] """
    value = str(value).replace(" ", "").split(",")
    ret = []
    for p in filter(None, value):
        if "-" in p:
            s, e = map(int, p.split("-"))
            ret.extend(range(s, e + 1))
        else:
            ret.append(int(p))
    return ret


def parse_txt_cfg(file, encoding="utf-8", comments="#"):
    """
    Supported syntax:
        content     # comment
    """
    with open(file, encoding=encoding) as f:
        for i_line in enumerate(s.split(comments)[0].strip() for s in f.read().splitlines()):
            if i_line[1]: yield i_line


def as_config(config):
    """ Convert input to a configuration. """
    if not isinstance(config, (dict, list)):
        config = Path(config)
        assert config.suffix[1:] in ("json", "yaml", "yml"), f"Unsupported config file format: {config.suffix}"
        config = config.f_load_dump()
    return config


def cfg2ns(config) -> list | SimpleNamespace:
    """ Convert input configuration to a namespace. """
    if isinstance(config, (list, tuple)):
        return [cfg2ns(x) for x in config]
    elif isinstance(config, dict):
        config = {k: cfg2ns(v) for k, v in config.items()}
        return SimpleNamespace(**config)
    else:
        return config
