from __future__ import absolute_import
from imgui.integrations.pygame import PygameRenderer
import OpenGL.GL as gl
from OpenGL.GL import (
    GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
    GL_NEAREST, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_CLAMP, GL_RGB, GL_UNSIGNED_BYTE, GL_LIGHTING,
    GL_QUADS, GL_PROJECTION, GL_MODELVIEW, GL_SMOOTH,
    GL_DEPTH_TEST, GL_PERSPECTIVE_CORRECTION_HINT, GL_LEQUAL,
    GL_NICEST, GL_BLEND
)
import imgui
import pygame
import sys
import os
import logging
from datetime import datetime
import pickle

import depth
from depth import ModelTriple
import calibrate

from drone import Drone
import drone_main

import numpy as np
import cv2 as cv

from djitellopy import Tello
import threading
import time


WIDTH, HEIGHT = 640, 480


def get_depth_map_from_matlike(frame: cv.typing.MatLike, model: ModelTriple) -> np.ndarray:
    """
    Generates a depth map from an OpenCV2 MatLike and applies `cv.COLORMAP_MAGMA` for depth visualization.
    """

    depth_img = depth.estimate(np.array([frame]), model)
    depth_img = depth.normalize(depth_img) # you can get a sick AMV effect when you comment this out
    return depth_img


def ndarray_to_pygame_surface(image: np.ndarray, size: tuple[int, int]=(WIDTH, HEIGHT)) -> pygame.Surface:
    """
    Parses an opencv value matrix as a renderable pygame surface
    You can get a cool effect if you skip this step of the render pipeline
    """

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = np.rot90(image)
    image = np.flipud(image)

    pygame_surface = pygame.surfarray.make_surface(image)
    pygame_surface = pygame.transform.scale(pygame_surface, size)
    return pygame_surface


def send_surface_to_texbuffer(pygame_surface, tex_id):
    """
    Sends a pygame surface (normally the window buffer) to a texture buffer `tex_id`.
    """

    rgb_surface = pygame.image.tostring(pygame_surface, 'RGB')
    
    gl.glBindTexture(GL_TEXTURE_2D, tex_id)
    
    surface_rect = pygame_surface.get_rect()
    
    gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, surface_rect.width, surface_rect.height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_surface)
    gl.glGenerateMipmap(GL_TEXTURE_2D)
    gl.glBindTexture(GL_TEXTURE_2D, 0)


def gl_init():
    """
    Initialize GL stuff
    """

    info = pygame.display.Info()
    gl.glViewport(0, 0, info.current_w, info.current_h)
    gl.glDepthRange(0, 1)
    gl.glMatrixMode(GL_PROJECTION)
    gl.glMatrixMode(GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.glShadeModel(GL_SMOOTH)
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glClearDepth(1.0)
    gl.glDisable(GL_DEPTH_TEST)
    gl.glDisable(GL_LIGHTING)
    gl.glDepthFunc(GL_LEQUAL)
    gl.glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
    gl.glEnable(GL_BLEND)
    gl.glDisable(GL_LIGHTING)
    gl.glEnable(GL_TEXTURE_2D)


def draw_pygame_window_buffer(window: pygame.Surface, tex_id: gl.GL_UNSIGNED_INT):
    """
    Draws the `window` pygame buffer in the current OpenGL context.
    
    OpenGL is called directly because the imgui renderer overrides the pygame window buffer,
    which must be passed manually to a GPU buffer (the texture buffer).
    """

    gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    send_surface_to_texbuffer(window, tex_id)
    gl.glBindTexture(GL_TEXTURE_2D, tex_id)

    gl.glLoadIdentity()
    gl.glBegin(GL_QUADS)

    gl.glTexCoord2f(0, 0); gl.glVertex2f(-1, 1)
    gl.glTexCoord2f(0, 1); gl.glVertex2f(-1, -1)
    gl.glTexCoord2f(1, 1); gl.glVertex2f(1, -1)
    gl.glTexCoord2f(1, 0); gl.glVertex2f(1, 1)

    gl.glEnd()


def imgui_init(size: tuple[int, int]=(WIDTH, HEIGHT)) -> PygameRenderer:
    """
    Initialized ImGUI and returns the backend renderer wrapper.
    """

    imgui.create_context()
    impl = PygameRenderer()

    io = imgui.get_io()
    io.display_size = size

    return impl


def handle_window_events(impl: PygameRenderer):
    """
    Handles pygame window events and passes them through the imgui PygameRenderer wrapper.
    """

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit(0)
        impl.process_event(event)
    impl.process_inputs()


class InfoFrameData:
    def __init__(self, depth_img: np.ndarray):
        self.depth_img = depth_img


def draw_info_frame(impl: PygameRenderer, info_frame_data: InfoFrameData):
    """
    Draws imgui frames.
    """

    imgui.new_frame()

    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File", True):

            clicked_quit, _ = imgui.menu_item(
                "Quit", "Cmd+Q", False, True
            )

            if clicked_quit:
                sys.exit(0)

            imgui.end_menu()
        imgui.end_main_menu_bar()

    imgui.begin("Depth Information")
    imgui.text(f"Mean: {round(np.mean(info_frame_data.depth_img) / 255, 2)}")
    imgui.end()

    imgui.render()
    impl.render(imgui.get_draw_data())


class DepthRenderContext:
    def __init__(self, window_title: str="Depth Render Demo", model: depth.Model=depth.Model.Small, size: tuple[int, int]=(WIDTH, HEIGHT)):
        self.model = ModelTriple.from_model(model)
        self.size = size

        pygame.init()
        self.window = pygame.display.set_mode(self.size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
        pygame.display.set_caption(window_title)

        gl_init()
        
        self.m_tex_id = gl.glGenTextures(1)
        gl.glBindTexture(GL_TEXTURE_2D, self.m_tex_id)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        gl.glBindTexture(GL_TEXTURE_2D, 0)

        self.m_renderer = imgui_init()
    
    # def __del__(self):
    #     if self.model != None:
    #         self.quit()
    
    def quit(self):
        gl.glBindTexture(GL_TEXTURE_2D, 0)
        gl.glDeleteTextures(1, self.m_tex_id)

        self.m_renderer.shutdown()
        pygame.quit()

        self.model = None
    
    def handle_events(self):
        handle_window_events(self.m_renderer)

    def draw_surface(self, surface: np.ndarray):
        self.window.blit(surface, (0, 0))
        draw_pygame_window_buffer(self.window, self.m_tex_id)
    
    def draw_ui(self, info: InfoFrameData):
        draw_info_frame(self.m_renderer, info)
    
    def swap_buffers(self):
        pygame.display.flip()


collisions_caught = 0
def log_collision(mean: float, frame: cv.typing.MatLike, depth_img: np.ndarray, postfix="", collision_id=0):
    global collisions_caught
    timefmt = datetime.now().strftime("%m-%d-%Y-%H-%M-%S.%f")[:-3]
    filepath = f"logs/images/depth-{collision_id}-{timefmt}-{postfix}.png"
    rawimgpath = f"logs/rawimg/rawimg-{collision_id}-{timefmt}-{postfix}.jpg"
    dumppath = f"logs/dumps/pickle-dump-{collision_id}-{timefmt}-{postfix}.bin"
    
    collisions_caught += 1
    
    logging.getLogger(__name__).warning(f"[{timefmt}] Collision threshold reached. Collisions caught: {collisions_caught}. Logging...")

    with open("logs/collision.log", "a") as f:
        f.write(f"[{timefmt}] mean={mean}:{filepath}:{dumppath}:{rawimgpath},\n")
    
    with open(dumppath, "wb") as f:
        pickle.dump(depth_img.tobytes(), f)
    
    cv.imwrite(filepath, depth_img)
    cv.imwrite(rawimgpath, frame)


COLLISION_THRESHOLD_PERCENTAGE = 0.50


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


def main():
    context = DepthRenderContext(model=depth.Model.Small)
    os.makedirs(os.path.dirname("logs/images/"), exist_ok=True)
    os.makedirs(os.path.dirname("logs/rawimg/"), exist_ok=True)
    os.makedirs(os.path.dirname("logs/dumps/"), exist_ok=True)

    # start drone
    drone = Drone()
    drone.connect()

    SNAPSHOT_DURATION_SECONDS = 3.0

    print("Starting drone sequence in five seconds. Make sure drone is at intended start position.")
    time.sleep(5)
    print("Starting drone sequence...")
    time.sleep(1)

    # Drone movement script here
    drone.tello.connect()
    drone.tello.streamon()
    seq_thread = StoppableThread(target=drone_main.start, args=(drone,))
    seq_thread.start()
    # end

    while True:
        context.handle_events()

        frame = drone.tello.get_frame_read().frame
        if frame is None:
            break

        depth_img = get_depth_map_from_matlike(frame, context.model)
        if depth_img is None:
            raise Exception('Could not generate depth map of scene.')
        
        surface = ndarray_to_pygame_surface(depth_img)
        if surface is None:
            raise Exception('Could not convert numpy array to pygame surface object.')
        
        context.draw_surface(surface)
        context.draw_ui(InfoFrameData(depth_img))

        mean = round(np.mean(depth_img) / 255, 2)
        # if mean is above threshold:
            # save snapshot of current frame
            # tell drone to stop
            # wait three seconds
            # save snapshot of current frame
        
        is_logging = getattr(main, "is_logging", False)
        time_start = getattr(main, "logging_time_start", 0)
        collision_id = getattr(main, "collision_id", 1)
        time_now = time.time()

        if is_logging:
            elapsed = time_now - time_start
            
            if elapsed >= SNAPSHOT_DURATION_SECONDS:
                print("Logging final collision state.")
                log_collision(mean, frame, depth_img, f"D{SNAPSHOT_DURATION_SECONDS}_final", collision_id) # save final snapshot
                exit(0)
                is_logging = False
        
        elif mean > COLLISION_THRESHOLD_PERCENTAGE:
            print("Initial collision detected.")
            collision_id += 1
            log_collision(mean, frame, depth_img, f"D{SNAPSHOT_DURATION_SECONDS}_initial", collision_id) # save initial snapshot
            
            time_start = time.time()
            seq_thread.stop()
            drone_main.stop(drone)
            is_logging = True
        
        setattr(main, "is_logging", is_logging)
        setattr(main, "logging_time_start", time_start)
        setattr(main, "collision_id", collision_id)

        # if mean > COLLISION_THRESHOLD_PERCENTAGE:
            

        #     if is_logging:
               
        #        time_now = time.time()
        #        elapsed = time_now - time_start

        #        if elapsed >= SNAPSHOT_DURATION_SECONDS:
        #            ... # save final snapshot
        #            is_logging = False
            
        #     else:
        #         is_logging = True
        #         time_start = time.time()

        #         ... # save initial snapshot

        


        # if mean > COLLISION_THRESHOLD_PERCENTAGE:
        #     last_log_time = getattr(main, "last_log_time", None)

        #     if last_log_time is None:
        #         last_log_time = datetime.fromtimestamp(0)
        #         setattr(main, "last_log_time", last_log_time)

        #     time_since_last_log = datetime.now() - last_log_time
        #     if time_since_last_log.total_seconds() >= 1:
        #         log_collision(mean, frame, depth_img)
        #         setattr(main, "last_log_time", datetime.now())

        context.swap_buffers()

    exit(0)


if __name__ == "__main__":
    main()
