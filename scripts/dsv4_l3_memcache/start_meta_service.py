import json
import logging
import sys

from memcache_hybrid import MetaConfig, MetaService


def main(path: str) -> None:
    with open(path, encoding="utf-8") as file:
        values = json.load(file)
    config = MetaConfig()
    for key, value in values.items():
        setattr(config, key, value)
    result = MetaService.setup(config)
    if isinstance(result, int) and result != 0:
        raise RuntimeError(f"MetaService.setup failed: {result}")
    MetaService.main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1])
