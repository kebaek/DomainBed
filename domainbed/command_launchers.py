# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """Doesn't run anything; instead, prints each command.
    Useful for testing."""
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def parallel_launcher(commands):
    child_processes = []
    for cmd in commands:
        p = subprocess.Popen(cmd, shell=True)
        child_processes.append(p)

    for cp in child_processes:
        cp.wait()

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'parallel': parallel_launcher
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
