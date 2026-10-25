"""Fresh-process smoke: optional ML packages absent, network blocked, all modules import."""

import importlib
import importlib.abc
import importlib.util
import json
import pkgutil
import platform
import socket
import sys

optional = {name: importlib.util.find_spec(name) is not None for name in ("sdv", "torch")}
assert not any(optional.values()), f"smoke requires an environment without SDV/torch: {optional}"


class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in optional:
            raise AssertionError(f"unexpected optional ML import: {fullname}")
        return None


def block_network(*args, **kwargs):
    raise AssertionError("network forbidden during import smoke")


sys.meta_path.insert(0, BlockOptional())
socket.create_connection = block_network
socket.getaddrinfo = block_network
for method in ("connect", "connect_ex", "sendto", "sendmsg"):
    if hasattr(socket.socket, method):
        setattr(socket.socket, method, block_network)

package = importlib.import_module("tabgen_bench")
modules = sorted(m.name for m in pkgutil.walk_packages(package.__path__, package.__name__ + "."))
for module in modules:
    importlib.import_module(module)
assert not any(name.split(".", 1)[0] in optional for name in sys.modules)
print(
    json.dumps(
        {
            "python": platform.python_version(),
            "sdv_installed": optional["sdv"],
            "torch_installed": optional["torch"],
            "network": "blocked",
            "imported_modules": modules,
        },
        sort_keys=True,
    )
)
