from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Package:
    source: Path
    dependency: dict[str, list[str]] = field(default_factory=dict)
    packages: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.source = Path(self.source)

    def collect_require(self, require: list[str]) -> list[str]:
        require = require.copy()

        # collect dependency
        i = 0
        while i < len(require):
            for dep in self.dependency.get(require[i], []):
                if dep not in require: require.append(dep)
            i += 1

        # resolve path
        for i, r in enumerate(require):
            f = self.source / (r + "/__init__" * (r in self.packages))
            f = f if f.exists() else f.with_suffix(".py")
            assert f.exists(), f"Dependency '{r}' not found at '{f}'."
            require[i] = str(f.relative_to(self.source))

        init = self.source / "__init__.py"
        if init.exists():
            require.append(init.name)
        return sorted(require)

    def install(self,
                dst: Path,
                require: list[str],
                collect: bool = True,
                empty_init: bool = False) -> Path:
        if collect: require = self.collect_require(require)

        dst = Path(dst)
        dst = dst / self.name if not dst.name == self.name else dst
        # input(f"Press Enter to install '{self.name}' to '{dst.resolve()}': ")

        for p in require:
            s, d = (self.source / p).resolve(), (dst / p).resolve()
            if s.is_file():
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(s, d)
                if empty_init and p.endswith("__init__.py"): d.write_text("")
            else:
                shutil.copytree(s, d, dirs_exist_ok=True)
        return dst

    @property
    def name(self) -> str:
        return self.source.name
