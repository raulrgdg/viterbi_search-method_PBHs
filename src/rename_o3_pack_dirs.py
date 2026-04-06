import argparse
from pathlib import Path

from pipeline_paths import INPUTS_O3_DATA_DIR


def rename_o3_pack_dirs(base_dir: Path) -> list[tuple[Path, Path]]:
    renamed: list[tuple[Path, Path]] = []

    if not base_dir.exists():
        raise FileNotFoundError(f"No existe el directorio: {base_dir}")

    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue
        if not path.name.startswith("O3b-pack") or not path.name.endswith("-512HZ"):
            continue

        target = path.with_name(path.name.removesuffix("-512HZ"))
        if target.exists():
            raise FileExistsError(f"El destino ya existe: {target}")

        path.rename(target)
        renamed.append((path, target))

    return renamed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Renombra carpetas O3b-packX-512HZ a O3b-packX."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=INPUTS_O3_DATA_DIR,
        help="Directorio base que contiene las carpetas O3b-packX-512HZ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renamed = rename_o3_pack_dirs(args.base_dir.resolve())

    if not renamed:
        print(f"No se encontraron carpetas para renombrar en {args.base_dir.resolve()}")
        return

    for source, target in renamed:
        print(f"{source.name} -> {target.name}")

    print(f"Renombradas {len(renamed)} carpeta(s) en {args.base_dir.resolve()}")


if __name__ == "__main__":
    main()
