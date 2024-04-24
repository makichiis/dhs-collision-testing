import drone

async def start(drone: drone.Drone):
    """ Begin the drone movement sequence. Executes asynchronously from the
        testing code, so do whatever you want. """
    drone.tello.connect()
    drone.tello.rotate_clockwise(90)
    drone.tello.set_speed(20)
