from Jumping_bird import *
import time
import random
from parameters import *
import numpy as np

# Store initial time
t0 = time.time()

# Get a random seed
random.seed(t0)

screen = Graphics()

bird = Bird(r=10, g=300)

running = True
jump = False
game_over = False
score = 0

# Create a lis of 3 pipe equally spaced
pipes = [Obstacle(PIPE_X0 + 2*ORIZONTAL_GAP*i, t0) for i in range(PIPE_ON_SCREEN)]

while running :

    running, jump = screen.user_interaction()

    bird.update(jump)

    for pipe in pipes:
        pipe.update()

        if pipe.x < - pipe.width:
            pipe.reset()

    if collision_detect(bird, pipes):
        print("Collison -> Game Over")
        break

    # Increase score
    _, nearest_pipe, score = distance_score(bird, pipes, score)


    entering_pipe(bird,pipes, nearest_pipe) 

    screen.draw(bird, pipes, 0, score, 0) 