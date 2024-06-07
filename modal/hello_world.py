"""
Running our function locally, remotely, and in parallel.
Source: https://modal.com/docs/examples/hello_world

To run:
>modal run hello_world.py

Modal’s goal is to make running code in the cloud feel like you’re running code locally. 
That means no waiting for long image builds when you’ve just moved a comma, 
no fiddling with container image pushes, and no context-switching to a web UI to inspect logs.
"""

import sys
import modal


app = modal.App(
    "example-hello-world"
)  # Note: prior to April 2024, "app" was called "stub"


@app.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i


@app.local_entrypoint()
def main():
    # run the function locally
    print(f.local(1000))

    # run the function remotely on Modal
    print(f.remote(1000))

    # run the function in parallel and remotely on Modal
    total = 0
    for ret in f.map(range(20)):
        total += ret

    print(total)
