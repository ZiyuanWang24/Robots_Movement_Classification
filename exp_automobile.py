__credits__ = ["Andrea PIERRÉ"]

import math
from typing import Optional, Union

import numpy as np
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import time
import colorama

from colorama import Fore,Style,Back

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle
import joblib

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError:
    raise DependencyNotInstalled("box2D is not installed, run `pip install gym[box2d]`")

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gym[box2d]`"
    )


STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 20.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 1  # Camera zoom
ZOOM_FOLLOW = False  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40*100 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap = True
        else:
            obj.tiles.remove(tile)


class CarRacing(gym.Env, EzPickle):
    """
    ### Description
    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```
    python gym/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ### Action Space
    If continuous:
        There are 3 actions: steering (-1 is full left, +1 is full right), gas, and breaking.
    If discrete:
        There are 5 actions: do nothing, steer left, steer right, gas, brake.

    ### Observation Space
    State consists of 96x96 pixels.

    ### Rewards
    The reward is -0.1 every frame and +1000/N for every track tile visited,
    where N is the total number of tiles visited in the track. For example,
    if you have finished in 732 frames, your reward is
    1000 - 0.1*732 = 926.8 points.

    ### Starting State
    The car starts at rest in the center of the road.

    ### Episode Termination
    The episode finishes when all of the tiles are visited. The car can also go
    outside of the playfield - that is, far off the track, in which case it will
    receive -100 reward and die.

    ### Arguments
    `lap_complete_percent` dictates the percentage of tiles that must be visited by
    the agent before a lap is considered complete.

    Passing `domain_randomize=True` enables the domain randomized variant of the environment.
    In this scenario, the background and track colours are different on every reset.

    Passing `continuous=False` converts the environment to use discrete action space.
    The discrete action space has 5 actions: [do nothing, left, right, gas, brake].

    ### Reset Arguments
    Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
    Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
    `domain_randomize` must be `True` on init for this argument to work.
    Example usage:
    ```py
        env = gym.make("CarRacing-v1", domain_randomize=True)

        # normal reset, this changes the colour scheme by default
        env.reset()

        # reset with colour scheme change
        env.reset(options={"randomize": True})

        # reset with no colour scheme change
        env.reset(options={"randomize": False})
    ```

    ### Version History
    - v1: Change track completion logic and add domain randomization (0.24.0)
    - v0: Original version

    ### References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

    ### Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
        )
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car: Optional[Car] = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )
        # Ziyuan Wang CSE583
        self.updis = 10
        self.downdis = 10
        self.leftdis = 10
        self.rightdis = 10

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, left, right, gas, brake

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.render_mode = render_mode

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        assert self.car is not None
        self.car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _reinit_colors(self, randomize):
        assert (
            self.domain_randomize
        ), "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    def _create_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        track.append((0, 0, 0, 0))
        track.append((0, 0, 0, 100))
        track.append((0, 0, 0, 200))
        track.append((0, 0, 0, 300))

        # Create tiles
        inner_boundary = []
        outer_boundary = []
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_only_poly.append([road1_l, road1_r, road2_r, road2_l])
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            inner_boundary.append(vertices[0])
            inner_boundary.append(vertices[3])
            outer_boundary.append(vertices[1])
            outer_boundary.append(vertices[2])
            '''
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )
        '''
        self.track = track
        self.track_boundary = [inner_boundary, outer_boundary]

        # # gzy: store track as a binary mask
        # track_bwmask = np.zeros((int(PLAYFIELD * 2), int(PLAYFIELD * 2)))
        # for poly in env.road_poly:
        #     pts_float = np.round(np.array(poly[0])) + PLAYFIELD
        #     pts_float = np.array([[pt[1], pt[0]] for pt in pts_float])    # exchange row and column
        #     cv2.fillPoly(track_bwmask, pts = [pts_float.astype(int)], color = 1)
        # erode_kernel = np.ones((3, 3), np.uint8)
        # track_bwmask_erode1 = cv2.erode(track_bwmask, erode_kernel, iterations=1)
        # self.track_bwmask = track_bwmask_erode1.astype(bool)


        return True

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
        self.road_only_poly = []

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        self.car = Car(self.world, *self.track[0][1:4])

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(self, action: Union[np.ndarray, int]):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        step_reward = 0
        terminated = False
        truncated = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Truncation due to finishing lap
                # This should not be treated as a failure
                # but like a timeout
                truncated = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                step_reward = -100

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()

        if mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                self.surf, poly, self.grass_color, zoom, translation, angle
            )

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)


        # Ziyuan Wang Draw trjectories
        red = (255,0,0)
        distances = [self.updis, self.downdis, self.leftdis, self.rightdis]
        

        # traj_reward = self.get_reward(T_arc_poly, T_func).astype(int)
        c_x, c_y = self.car.hull.position
        c_angle = self.car.hull.angle
        center_point = [c_x, c_y]
        up_point = [c_x - 10*self.updis * np.sin(c_angle), c_y + 10*self.updis * np.cos(c_angle)]
        down_point = [c_x + 10*self.downdis * np.sin(c_angle), c_y - 10*self.downdis * np.cos(c_angle)]
        left_point = [c_x - 10*self.leftdis * np.cos(c_angle), c_y - 10*self.leftdis * np.sin(c_angle)]
        right_point = [c_x + 10*self.rightdis * np.cos(c_angle), c_y + 10*self.rightdis * np.sin(c_angle)]
        points = [up_point, down_point, left_point, right_point, center_point]
        for i in range(4):           
            self._draw_colored_line(self.surf, center_point, points[i], red, zoom, translation, angle, distances[i])

    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    # Ziyuan Wang: draw line
    def _draw_colored_line(
        self, surface, point1, point2, color, zoom, translation, angle, Num = [], clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in [point1, point2]]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.line(surface, int(poly[0][0]), int(poly[0][1]), int(poly[1][0]), int(poly[1][1]), color)

        font = pygame.font.Font(pygame.font.get_default_font(), 10)
        # text = font.render("%04i" % Num, True, (255, 255, 255), (0, 0, 0))
        text = font.render(str(Num), True, (255, 255, 255), (0, 0, 0))
        text = pygame.transform.rotate(text, 180)
        text = pygame.transform.flip(text, True, False)
        text_rect = text.get_rect()
        poly_count_half = np.round(len(poly) / 2).astype(int)
        text_rect.center = (poly[poly_count_half][0], poly[poly_count_half][1])
        self.surf.blit(text, text_rect)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()

    def gen_Trajectories(self, K, arcLength):
        '''
        Input:
        K: numbers of generated trajectories
        arcLength: the lenth of each arc
        Output:
        trajs_funcs: K trajectories functions. 
        K * 4: K * [a, b, r, theta_max]
        '''
        x, y = self.car.hull.position
        angle = self.car.hull.angle
        half_K = int(K/2)
        Rs = np.array([-100**(power+1) for power in range(half_K)]
                        + [+100**(power+1) for power in range(half_K)])
        Rs = np.linspace(-half_K*100, half_K*100 , K)
        trajs_funcs = np.array([[x+r*np.cos(angle), y+r*np.sin(angle), abs(r), arcLength/abs(r)] for r in Rs])

        return trajs_funcs

    # gzy: get the reward of a specific trajectory
    def get_reward(self, traj_samples, T_func):
        reward = 0

        a = T_func[0]
        b = T_func[1]
        R = T_func[2]
        arc0 = self.car.hull.angle
        arc1 = arc0 + T_func[3]
        velocity = 10
        within_track = True
        idx_sample = 0

        while within_track:
            traj_pt = traj_samples[idx_sample]

            inner_path = mpltPath.Path(self.track_boundary[0])
            outer_path = mpltPath.Path(self.track_boundary[1])
            pt_within_track = (not inner_path.contains_point(traj_pt)) & outer_path.contains_point(traj_pt)

            within_track &= pt_within_track
            idx_sample += 1

            if idx_sample == len(traj_samples):
                idx_sample -= 1
                break
            if not within_track:
                idx_sample -= 2
                if idx_sample < 0:
                    idx_sample = 0
                break

        if not within_track:
            reward -= 100

        # print(idx_sample)
        # print(len(traj_samples)-1)

        # Find the tile that the car is currently in, starting from the tile_visited_count and searching forward
        current_tile_num = self.tile_visited_count % len(self.track) - 1
        if current_tile_num < 0:
            current_tile_num = 0
        path = mpltPath.Path(self.road_only_poly[current_tile_num])
        start_pt = np.array(self.car.hull.position)
        inside_tile = path.contains_point(start_pt)
        while not inside_tile:
            current_tile_num += 1
            path = mpltPath.Path(self.road_only_poly[current_tile_num])
            inside_tile = path.contains_point(start_pt)

        # Find the tile that the terminal of the trajectory is in, searching forward from the current tile num
        term_point = traj_samples[idx_sample]
        terminal_tile_num = current_tile_num
        inside_tile = path.contains_point(term_point)
        while not inside_tile:
            terminal_tile_num += 1
            path = mpltPath.Path(self.road_only_poly[terminal_tile_num])
            inside_tile = path.contains_point(term_point)

            # print(path)
            # print(term_point)

        # Update reward by the number of tiles to be visited
        reward += 1000 * (terminal_tile_num - current_tile_num) / len(self.track)

        # Update penalty by the time elapsed
        reward -= R * (arc1 - arc0) / velocity * 0.1 * FPS

        return reward

if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])
    restart = False

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = CarRacing(render_mode="human")
    env.reset()

    def go_straight(speed):
        true_speed = np.sqrt(
                np.square(env.car.hull.linearVelocity[0])
                + np.square(env.car.hull.linearVelocity[1])
            )
        while true_speed < speed:
            env.step(np.array([0,1,0]))
            true_speed = np.sqrt(
                np.square(env.car.hull.linearVelocity[0])
                + np.square(env.car.hull.linearVelocity[1])
            )

    def turn_left():
        print('Turn Left')
        steps = 0
        turn_step = 1
        while env.car.hull.angle < np.pi/2:
            a[0] = 0.0
            a[1] = 0.0
            a[2] = 0.0
            
            if steps % turn_step == 0:
                a[0] = -1.0
            s, r, terminated, truncated, info = env.step(a)
            steps += 1
            time.sleep(0.1)
        a[0] = 0
        a[2] = 1.0
        time.sleep(3)

    def turn_right():
        print('Turn Right')
        steps = 0
        turn_step = 1
        while env.car.hull.angle > -np.pi/2:
            a[0] = 0.0
            a[1] = 0.0
            a[2] = 0.0
            
            if steps % turn_step == 0:
                a[0] = 1.0
            s, r, terminated, truncated, info = env.step(a)
            steps += 1
            time.sleep(0.1)
        a[0] = 0
        a[2] = 1.0
        time.sleep(3)



    env_continue = True
    turn_end = True
    # while env_continue:
    #     if turn_end:
    #         _env_continue = input('Continue (True or False): ')
    def determineDir():

        colorama.init()

        YELLOW = "\x1b[1;33;40m" 
        RED = "\x1b[1;31;40m"
        CYAN = '\033[36m'
        MAGENTA = '\033[35m'

        model_select = input(f'\n{CYAN}Model (GNB, NN, SVM, LR): ')
        if model_select == 'GNB':
            model = joblib.load('src/saved_model/GaussianNB.pkl')
        elif model_select == 'SVM':
            model = joblib.load('src/saved_model/SVMcalssifier.pkl')
        elif model_select == 'NN':
            model = joblib.load('src/saved_model/NeuralNet.pkl')
        elif model_select == 'LR':
            model = joblib.load('src/saved_model/linear_model.pkl')
        else:
            print(f'\n{CYAN}Default: NN')
            model = joblib.load('src/saved_model/NeuralNet.pkl')
        
        # print(model.predict(np.array([[1.687, 0.445, 2.332, 0.429]])))
        env.updis = up_dis = float(input(f'\n{MAGENTA}Up Distance: '))
        env.leftdis = left_dis = float(input(f'\n{MAGENTA}Left Distance: '))
        env.rightdis = right_dis = float(input(f'\n{MAGENTA}Right Distance: '))
        env.downdis = down_dis = float(input(f'\n{MAGENTA}Down Distance: '))
        dir = model.predict(np.array([[up_dis, left_dis, right_dis, down_dis]]))
        print(dir)
        return dir
    dir = determineDir()
    # dir = model.predict(np.array([[1.687, 0.445, 2.332, 0.429]]))
    go_straight(10)

    quit = False
    while True:
        if dir == 'Slight-Right-Turn' or dir == 'Sharp-Right-Turn':
            turn_right()
        elif dir == 'Slight-Left-Turn':
            turn_left()
        elif dir == 'Move-Forward':
            go_straight(30)

        env.reset()
        
        dir = determineDir()
        RED = "\x1b[1;31;40m"
        stop = input(f'\n{RED}Stop (True or False): ')
        if stop == 'True':
            break
        go_straight(10)

    env.close()