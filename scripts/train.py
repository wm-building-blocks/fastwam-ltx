# faulthandler: when SIGSEGV/SIGFPE/SIGABRT/SIGBUS fires, dump every thread's
# Python + C stack to stderr before the process dies. Zero overhead while
# running; gives a traceback when a native crash (NCCL, CUDA, decode) would
# otherwise leave no Python stack. SIGUSR1 dumps without killing (debug hangs).
import faulthandler
import signal
import sys

faulthandler.enable(file=sys.stderr, all_threads=True)
try:
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
except (AttributeError, ValueError):
    pass

import hydra
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
