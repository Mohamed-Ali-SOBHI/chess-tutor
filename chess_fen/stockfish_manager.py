import glob
import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


DOWNLOAD_VARIANTS = [
    {
        "slug": "avx2",
        "label": "AVX2",
        "url": "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.zip",
    },
    {
        "slug": "popcnt",
        "label": "SSE4.1 POPCNT",
        "url": "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-sse41-popcnt.zip",
    },
    {
        "slug": "x64",
        "label": "x86-64",
        "url": "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64.zip",
    },
]


def _project_root():
    return Path(__file__).resolve().parents[1]


def _managed_stockfish_root():
    return _project_root() / "runtime" / "stockfish"


def _normalize_candidate_path(candidate_path):
    if not candidate_path:
        return None

    expanded_path = os.path.expandvars(os.path.expanduser(str(candidate_path)))
    normalized_path = os.path.abspath(expanded_path)
    if os.path.isfile(normalized_path):
        return normalized_path
    return None


def _find_stockfish_executable(search_root):
    search_root = Path(search_root)
    if not search_root.exists():
        return None

    matches = sorted(
        search_root.rglob("stockfish*.exe"),
        key=lambda path: (len(path.parts), len(path.name)),
    )
    for match in matches:
        if match.is_file():
            return str(match.resolve())
    return None


def _iter_common_installation_candidates():
    project_root = _project_root()
    path_patterns = [
        project_root / "stockfish.exe",
        project_root / "stockfish" / "stockfish.exe",
        project_root / "engines" / "stockfish.exe",
    ]

    for path_pattern in path_patterns:
        normalized_path = _normalize_candidate_path(path_pattern)
        if normalized_path:
            yield normalized_path

    glob_patterns = [
        str(project_root / "stockfish" / "**" / "stockfish*.exe"),
        str(project_root / "engines" / "**" / "stockfish*.exe"),
        str(_managed_stockfish_root() / "**" / "stockfish*.exe"),
    ]

    env_roots = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("LOCALAPPDATA"),
        os.environ.get("ChocolateyInstall"),
    ]
    for env_root in env_roots:
        if not env_root:
            continue

        glob_patterns.extend(
            [
                os.path.join(env_root, "Stockfish", "**", "stockfish*.exe"),
                os.path.join(env_root, "Programs", "Stockfish", "**", "stockfish*.exe"),
                os.path.join(env_root, "lib", "stockfish*", "**", "stockfish*.exe"),
            ]
        )

    for pattern in glob_patterns:
        for match in sorted(glob.glob(pattern, recursive=True)):
            normalized_path = _normalize_candidate_path(match)
            if normalized_path:
                yield normalized_path


def iter_existing_stockfish_paths(stockfish_path=None):
    seen_paths = set()

    def _yield_candidate(candidate_path):
        normalized_path = _normalize_candidate_path(candidate_path)
        if normalized_path and normalized_path not in seen_paths:
            seen_paths.add(normalized_path)
            return normalized_path
        return None

    explicit_path = _yield_candidate(stockfish_path)
    if explicit_path:
        yield explicit_path

    for env_var in ("STOCKFISH_PATH", "STOCKFISH_EXECUTABLE"):
        env_path = _yield_candidate(os.environ.get(env_var))
        if env_path:
            yield env_path

    which_stockfish = shutil.which("stockfish") or shutil.which("stockfish.exe")
    which_path = _yield_candidate(which_stockfish)
    if which_path:
        yield which_path

    for candidate_path in _iter_common_installation_candidates():
        normalized_path = _yield_candidate(candidate_path)
        if normalized_path:
            yield normalized_path


def _download_stockfish_variant(variant, progress_callback=None):
    variant_dir = _managed_stockfish_root() / variant["slug"]
    existing_executable = _find_stockfish_executable(variant_dir)
    if existing_executable:
        return existing_executable

    if progress_callback:
        progress_callback(
            f"Stockfish absent. Installation automatique ({variant['label']})..."
        )

    shutil.rmtree(variant_dir, ignore_errors=True)
    variant_dir.mkdir(parents=True, exist_ok=True)
    archive_path = variant_dir / "stockfish.zip"

    request = Request(
        variant["url"],
        headers={"User-Agent": "ChessScreenAssistant/1.0"},
    )

    try:
        with urlopen(request, timeout=180) as response, open(archive_path, "wb") as file_obj:
            shutil.copyfileobj(response, file_obj)

        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(variant_dir)
    finally:
        if archive_path.exists():
            archive_path.unlink()

    executable_path = _find_stockfish_executable(variant_dir)
    if executable_path is None:
        raise RuntimeError("Archive Stockfish invalide: executable introuvable")

    return executable_path


def iter_installable_stockfish_paths(progress_callback=None):
    yielded_paths = set()

    for variant in DOWNLOAD_VARIANTS:
        existing_path = _find_stockfish_executable(_managed_stockfish_root() / variant["slug"])
        if existing_path and existing_path not in yielded_paths:
            yielded_paths.add(existing_path)
            yield existing_path

    for variant in DOWNLOAD_VARIANTS:
        installed_path = _find_stockfish_executable(_managed_stockfish_root() / variant["slug"])
        if installed_path:
            continue

        executable_path = _download_stockfish_variant(
            variant,
            progress_callback=progress_callback,
        )
        if executable_path not in yielded_paths:
            yielded_paths.add(executable_path)
            yield executable_path
