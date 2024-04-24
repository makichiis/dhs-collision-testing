import drone

# TODO: Specify drone height for separate test 
#        Ask Carter for meeting notes for types of tests to run

def start(drone: drone.Drone):
    """ Begin the drone movement sequence. Executes asynchronously from the
        testing code, so do whatever you want. """
    print("Starting drone movement sequence...")
    # drone.tello.connect()
    # drone.tello.takeoff()
    # drone.tello.rotate_clockwise(90)
    # drone.tello.set_speed(50)
    # drone.tello.move_forward(300)

def stop(drone: drone.Drone):
    """ End the drone movement sequence. """
    print("Ending drone movement sequence...")
#    drone.tello.land()

if __name__ == "__main__":
    d = drone.Drone()
    start(d)
    d.tello.end()
