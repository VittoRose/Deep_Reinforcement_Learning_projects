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

    dist = np.array([pipes[j].x for j in range(PIPE_ON_SCREEN)]) - np.array([bird.x for _ in range(PIPE_ON_SCREEN)])
    
    for i in range(len(dist)):

        if dist[i] <= -pipes[i].width:
            dist[i] = 1000
            if pipes[i].score_flag == True:
                score += 1
                print(f"Score: {score}")
                pipes[i].score_flag = False        

        elif dist[i] <= 0:
            dist[i] = 1000

    screen.draw(bird, pipes, 0, score)