# dhs-collision-testing
Collision testing repository for DHS-Spring2024

# Usage
> Python 3.11 REQUIRED

Install dependencies via `python -m pip -r requirements.txt`

Drone runtime logic is defined in `drone_main.py`, which is called in `collision_test.py`.

Execute `collision_test.py` with Python 3.11.

`drone_main.py` example code:
```py
import drone

async def start(drone: drone.Drone):
    """ Begin the drone movement sequence. Executes asynchronously from the
        testing code, so do whatever you want. """
    drone.tello.connect()
    drone.tello.rotate_clockwise(90)
    drone.tello.set_speed(20)
```