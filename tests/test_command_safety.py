"""Hardline command safety floor (INF-1): pori/sandbox/command_safety.py."""

import pytest

from pori.sandbox.command_safety import hardline_violation, normalize
from pori.sandbox.local import LocalSandbox

pytestmark = [pytest.mark.unit]

BLOCKED = [
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf ~/",
    "rm -rf $HOME",
    "sudo rm -fr /",
    "rm --no-preserve-root -rf /",
    "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda bs=1M",
    "echo x > /dev/sda",
    ":(){ :|:& };:",
    "shutdown -h now",
    "sudo reboot",
    "kill -9 -1",
    "kill -1",
    "make build | rm -rf /",
]

ALLOWED = [
    "rm -rf ./build",
    "rm -rf /tmp/scratch",
    "rm -rf ~/projects/pori/node_modules",
    "rm file.txt",
    "ls -la /",
    "echo shutdown the app when done",
    "echo mkfs is a tool",
    "git status",
    "python script.py",
    "pkill -f server",
    "dd if=input.bin of=output.bin",
]


@pytest.mark.parametrize("cmd", BLOCKED)
def test_hardline_blocks(cmd):
    assert hardline_violation(cmd) is not None, f"should block: {cmd!r}"


@pytest.mark.parametrize("cmd", ALLOWED)
def test_hardline_allows(cmd):
    assert hardline_violation(cmd) is None, f"should allow: {cmd!r}"


def test_obfuscation_is_normalized_before_matching():
    assert hardline_violation("rm''  -rf  /") is not None  # empty-string token split
    assert hardline_violation(r"rm\ -rf\ /") is not None  # backslash escapes
    assert normalize(r"rm\ -rf\ /") == "rm -rf /"


def test_local_sandbox_refuses_hardline_without_running():
    sb = LocalSandbox()
    out = sb.execute_command("rm -rf /")
    assert "hardline safety floor" in out
    # A harmless command still runs normally (the floor is targeted, not blanket).
    assert "pori-ok" in sb.execute_command("echo pori-ok")
