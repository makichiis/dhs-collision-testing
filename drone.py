from djitellopy import Tello

class Drone:
    def __init__(self):
        """ Initialize a Tello instance managed by this context. """
        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

    def connect(self):
        """ Sends command ping to tello drone and begins video stream """
        self.tello.connect()
        self.tello.set_speed(self.speed)

        self.tello.streamoff()
        self.tello.streamon()