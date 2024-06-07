"""
Quickstart.
Good to test the connection.
 
Install:
>pip install modal
>python3 -m modal setup

The second command creates an API token by authenticating through your web browser. 
It will open a new tab, but you can close it when you are done.

To run:
>modal run get_started.py

"""

import modal

app = modal.App(
    "example-get-started"
)  # Note: prior to April 2024, "app" was called "stub"


@app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2


@app.local_entrypoint()
def main():
    print("the square is", square.remote(42))
