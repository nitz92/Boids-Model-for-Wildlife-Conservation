import random
import pyglet
import math
import csv
from time import sleep
from pyglet import clock
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np

from pyglet.gl import (
    Config,
    glEnable, glBlendFunc, glLoadIdentity, glClearColor,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_COLOR_BUFFER_BIT)

from pyglet.window import key
from .boid import Boid
from .attractor import Attractor
from .obstacle import Obstacle
from .poacher import Poacher
from .ranger import Ranger
from .drone import Drone

globvar = 0
ranger_glob = 0
glob_time = 0
boidvar = 0
poacher_glob = 0
watering_holes_dt = 0
drone_glob = 0

def create_random_boid(height, width):
    return Boid(
        position=[random.uniform(0, height), random.uniform(0, width)],
        bounds=[1280, 720],
        velocity=[random.uniform(-50.0, 50.0), random.uniform(-50.0, 50.0)],
        color=[1.0, 1.0, 1.0])

def get_child_position(parent_position, parent_child_separation):
    child_position = [None, None]
    child_position[0] = parent_position[0] + np.random.uniform(-parent_child_separation, parent_child_separation)
    child_position[1] = parent_position[1] + np.random.uniform(-parent_child_separation, parent_child_separation)
    #child_position[1] = np.random.uniform(parent_position[1]-parent_child_separation, parent_position[1]+parent_child_separation)
    return child_position

def create_child_boid(boids, parent_child_separation):
    boid_parent = boids[np.random.randint(len(boids))]
    return Boid(
        position = get_child_position(boid_parent.position, parent_child_separation),
        bounds=[1280, 720],
        velocity=[random.uniform(-50.0, 50.0), random.uniform(-50.0, 50.0)],
        color=[1.0, 1.0, 1.0])

def create_random_poacher(width, height):
    return Poacher(
        position=[random.uniform(0, width), random.uniform(0, height)],
        bounds=[width, height],
        velocity=[random.uniform(-150.0, 150.0), random.uniform(-150.0, 150.0)],
        color=[0.8,0.1,0.0])

def create_random_ranger(width, height):
    return Ranger(
        position=[random.uniform(0, width), random.uniform(0, height)],
        bounds=[width, height],
        velocity=[random.uniform(-200.0, 200.0), random.uniform(-200.0, 200.0)],
        color=[0.0,0.8,0.0])

def create_random_drone(width, height):
    return Drone(
        position=[random.uniform(0, width), random.uniform(0, height)],
        bounds=[width, height],
        velocity=[random.uniform(-200.0, 200.0), random.uniform(-200.0, 200.0)],
        color=[1.0, 0.9, 0.0])

def create_random_attractor(width, height):
    return Attractor(
        position=[random.uniform(0, width), random.uniform(0, height)])

def get_nearby_poachers(drone,poachers):
    x_drone = drone.position[0]
    y_drone = drone.position[1]
    nearby_poachers = []
    try:
        for poacher in poachers:
            x_poacher = poacher.position[0]
            y_poacher = poacher.position[1]

            is_seen = ((x_poacher-x_drone)**2 + (y_poacher-y_drone)**2) <= drone.range**2

            if is_seen:
                nearby_poachers.append(poacher)
        return nearby_poachers
    except:
        return None


def update_rangers(rangers,nearby_poachers):
    #change the direction of velocity of the Rangers
    if nearby_poachers == None:
        pass
    else:
        try:
            for poacher in nearby_poachers:
                x_poacher = poacher.position[0]
                y_poacher = poacher.position[1]
                distances = []
                for ranger in rangers:
                    x_ranger = ranger.position[0]
                    y_ranger = ranger.position[1]

                    distances.append((x_poacher-x_ranger)**2 + (y_poacher-y_ranger)**2)
                closest_ranger_index = np.argmin(distances)
                closest_ranger = rangers[closest_ranger_index]

                x_velocity_ranger = closest_ranger.velocity[0]
                y_velocity_ranger = closest_ranger.velocity[1]

                velocity_closest_ranger = np.sqrt(x_velocity_ranger**2 + y_velocity_ranger**2)*1.05

                x_closest_ranger = closest_ranger.position[0]
                y_closest_ranger = closest_ranger.position[1]

                delta_y = y_poacher - y_closest_ranger
                delta_x = x_poacher - x_closest_ranger

                angle_between_positions = np.arctan(delta_y/delta_x)

                closest_ranger.velocity[0] = np.cos(angle_between_positions)*velocity_closest_ranger
                closest_ranger.velocity[1] = np.sin(angle_between_positions)*velocity_closest_ranger
        except:
            pass


def get_window_config():
    platform = pyglet.window.get_platform()
    display = platform.get_default_display()
    screen = display.get_default_screen()

    template = Config(double_buffer=True, sample_buffers=1, samples=4)
    try:
        config = screen.get_best_config(template)
    except pyglet.window.NoSuchConfigException:
        template = Config()
        config = screen.get_best_config(template)

    return config

def run():
    show_debug = False
    show_vectors = False
    boids = []
    attractors = []
    obstacles = []
    poachers = []
    rangers = []
    drones = []
    gt= []
    prey_data = []
    test_time = []
    #calc_probability = []

    mouse_location = (0, 0)
    window = pyglet.window.Window(
        fullscreen= True,
        caption="Boids Simulation",
        config=get_window_config())

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def test_file():
        with open('species.csv', mode='w') as data_file:
            rows = csv.writer(data_file, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            rows.writerow(gt)

    def plot_graph():
        df = pd.DataFrame({
            'Time': [x[0] for x in gt],
            'Preys': [x[1] for x in gt],
            'Poachers': [x[2] for x in gt],
            'Rangers': [x[3] for x in gt],
            'Drones' :[x[4] for x in gt]
        })
        ax = plt.gca()
        plt.title('Population Chart')
        plt.ylabel('Population Growth')
        plt.xlabel('Time(seconds)')
        df.plot(kind='line',x='Time',y='Preys',color='blue',ax=ax)
        df.plot(kind='line',x='Time',y='Poachers',color='red',ax=ax)
        df.plot(kind='line',x='Time',y='Rangers',color='green',ax=ax)
        df.plot(kind='line',x='Time',y='Drones',color='orange',ax=ax)
        plt.legend()
        plt.show()

    no_of_boids = 30
    no_of_poachers = 7
    no_of_rangers = 4
    no_of_attractors = 1
    no_of_drones = 3
    for i in range(no_of_boids):
        boids.append(create_random_boid(window.width, window.height))

    for k in range(no_of_attractors):
        attractors.append(create_random_attractor(window.width, window.height))

    for l in range(no_of_rangers):
        rangers.append(create_random_ranger(window.width, window.height))

    for b in range(no_of_drones):
        drones.append(create_random_drone(window.width,window.height))

    def update(dt):
        global globvar
        global boidvar
        global ranger_glob
        global poacher_glob
        global glob_time
        global watering_holes_dt
        global drone_glob

        gt.append([glob_time, len(boids), len(poachers), len(rangers), len(drones)])

        globvar += dt
        poacher_glob += dt
        ranger_glob += dt
        glob_time += dt
        boidvar += dt
        watering_holes_dt += dt
        drone_glob += dt

        if(glob_time > 60):
            pyglet.app.exit()
            test_file()
            plot_graph()

        if(boidvar >=1.0):
            number_new_boids = int(np.random.random()*.10*len(boids))
            for i in range(number_new_boids):
                boids.append(create_child_boid(boids, parent_child_separation=min(window.width, window.height)*.03))
            boidvar = 0

        if(watering_holes_dt >= 20):
            attractors.append(create_random_attractor(window.width, window.height))
            watering_holes_dt = 0

        if (poacher_glob >= 5 and len(boids) > no_of_boids*0.5):
            poachers.append(create_random_poacher(window.width, window.height))
            poacher_glob = 0

        if(ranger_glob >= 3 and len(poachers) > 3):
            rangers.append(create_random_ranger(window.width, window.height))
            ranger_glob = 0

        if(drone_glob >= 2 and len(poachers) > 3):
            drones.append(create_random_drone(window.width, window.height))
            drone_glob = 0

        if((ranger_glob >= 5 and len(boids) > no_of_boids*1.5)):
            try:
                rangers.pop()
            except:
                pass
            ranger_glob = 0

        if((drone_glob >= 50 and len(boids) > no_of_boids*1.5)):
            try:
                drones.pop()
            except:
                pass
            drone_glob = 0

        for boid in boids:
            boid.update(dt, boids, attractors, poachers)
            if boid.visualise == False:
                boids.pop(boids.index(boid))

        for poacher in poachers:
            poacher.update(dt, poachers, boids, rangers)
            if poacher.visualise == False:
                poachers.pop(poachers.index(poacher))

        for ranger in rangers:
            ranger.update(dt, rangers, poachers)

        for drone in drones:
            drone.update(dt, drones, boids)
            nearby_poachers = get_nearby_poachers(drone,poachers)
            update_rangers(rangers, nearby_poachers)

    # schedule world updates as often as possible
    pyglet.clock.schedule(update)

    @window.event
    def on_draw():
        glClearColor(0.1, 0.1, 0.1, 1.0)
        window.clear()
        glLoadIdentity()

        for boid in boids:
            boid.draw(show_velocity=show_debug, show_view=show_debug, show_vectors=show_vectors)

        for attractor in attractors:
            attractor.draw()

        for poacher in poachers:
            poacher.draw(show_velocity=show_debug, show_view=show_debug, show_vectors=show_vectors)

        for ranger in rangers:
            ranger.draw(show_velocity=show_debug, show_view=show_debug, show_vectors=show_vectors)

        for drone in drones:
            drone.draw(show_velocity=show_debug, show_view=show_debug, show_vectors=show_vectors)

    @window.event
    def on_key_press(symbol, modifiers):

        if symbol == key.Q:
            pyglet.app.exit()
        elif symbol == key.V:
            nonlocal show_vectors
            show_vectors = not show_vectors
        elif symbol == key.A:       #Angle of View for the prey
            nonlocal show_debug
            show_debug = not show_debug
        elif symbol == key.W:       #Add resources to attract the prey towards them
            attractors.append(Attractor(position=mouse_location))

    @window.event
    def on_mouse_drag(x, y, *args):
        nonlocal mouse_location
        mouse_location = x, y

    @window.event
    def on_mouse_motion(x, y, *args):
        nonlocal mouse_location
        mouse_location = x, y

    pyglet.app.run()
