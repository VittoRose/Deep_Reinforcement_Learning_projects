import pygame
import random
import time
from parameters import *
import numpy as np

MIN_VEL = 30

class Graphics():
    """
    Class that implement the drawing of the bird and obstacles
    """    
    def __init__(self):
        
        # Colors definition
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 180, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)

        # Init max score
        self.max_score = 0

        # Windows parameters
        self.width, self.height = WIDTH, HEIGHT

        # Call to create a screen
        pygame.init()
        
        # Font
        self.font = pygame.font.SysFont('Arial', 28)
        pygame.font.init()
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Flappy bird AI')

        pygame.key.set_repeat(0)

    def draw(self, bird, pipes, n_res, score, rew ,FPS:int = 60) -> None:
        """
        Draw everything on the screen
        """

        # Handle max score
        if score > self.max_score:
            self.max_score = score

        self.screen.fill(self.WHITE)

        pygame.draw.circle(self.screen, self.RED, (bird.x, bird.y), bird.r)

        for pipe in pipes:
            # Top pipe
            pygame.draw.rect(self.screen, self.GREEN, (pipe.x, pipe.y1,  pipe.width, pipe.h1))
            # Bottom pipe
            pygame.draw.rect(self.screen, self.GREEN, (pipe.x, pipe.y2, pipe.width, pipe.h2))

            # Debug point
            pygame.draw.circle(self.screen, self.BLACK, (pipe.x, pipe.h1), 5)
            pygame.draw.circle(self.screen, self.BLACK, (pipe.x, pipe.y2), 5)

            pygame.draw.circle(self.screen, (180,0,0), pipe.top_entr, 5)
            pygame.draw.circle(self.screen, (180,0,0), pipe.bottom_entr, 5)

            # Debug line
            pygame.draw.line(self.screen, self.BLACK, (pipe.x, pipe.h1), pipe.top_entr)
            pygame.draw.line(self.screen, self.BLACK, (pipe.x, pipe.y2), pipe.bottom_entr)

        text1 = self.font.render(f"Reset: {n_res} Score: {score}",True, self.BLACK, self.WHITE)
        text2 = self.font.render(f"Max {self.max_score} Rew {rew}",True, self.BLACK, self.WHITE)
        
        self.screen.blit(text1, (400, self.height-60))
        self.screen.blit(text2, (400, self.height-30))

        pygame.display.flip()

        pygame.time.Clock().tick(FPS)

    def user_interaction(self) -> tuple[bool, bool]:
        """
        Method that handle the jump button and quit event
        """

        # Output variable
        go_on = True
        jump = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                go_on = False
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                    jump = True
        return go_on, jump


class Bird():
    """
    Class that implement the bird and take care of the position
    """

    def __init__(self, r, dt : float = 1/60, g : float = 9.81):
        
        # Circle dimension
        self.r = r

        # Initial state
        self.y = HEIGHT//2
        self.vy = 0
        self.x = 160

        # Gravity
        self.g = -g

        # Sample time
        self.dt = dt

        # Bounce coeff
        self.bc = 0.65

    def update(self, jump : bool = False) -> None:
        """
        Update the y coord of the bird according to y = y0 + 
        """
        # Acceleration if jump
        if jump:
            self.vy -= 200
        else:
            self.vy = self.vy - self.g*self.dt

        self.y = self.y + self.vy*self.dt

        # Offset for collision top/bottom
        offset = 2

        # Lower collision
        if self.y >= HEIGHT - self.r + offset:
            self.vy = - self.bc* self.vy
            if self.vy < MIN_VEL:
                self.y = HEIGHT-self.r + offset 
                self.vt = 0

        # Upper collision
        if self.y <= self.r + offset:
            self.vy = - self.bc* self.vy
            self.y = self.r + offset
            if self.vy < MIN_VEL:
                self.y = self.r + offset 
                self.vt = 0


class Obstacle():
    def __init__(self, x0 : int, t0 , dt : float = 1/60):

        # Sample time
        self.dt = dt

        # Starting time 
        self.t0 = t0

        # Pipe width
        self.width = PIPE_WIDTH

        # Pipe height 
        self.gap = 100                                          # Clearance between pipes
        self.h1 = random.randint(30, HEIGHT - 1.5*self.gap)
        self.h2 = HEIGHT - self.gap - self.h1

        # Drawing point
        self.x =  x0
        self.y1 = 0
        self.y2 = self.h1 + self.gap

        # Entring point
        self.top_entr = (self.x-X_ENTR, self.h1-Y_ENTR)
        self.bottom_entr = (self.x-X_ENTR, self.y2+Y_ENTR)
        
        # Pipe velocity
        self.vx = -60

        # Score flag
        self.score_flag = True

    def update(self):
        """
        Update obstacle variable that change over time
        """
        # Update position
        self.x = self.x + self.vx*self.dt

        # Update entrig point
        self.top_entr = (self.x-X_ENTR, self.h1-Y_ENTR)
        self.bottom_entr = (self.x-X_ENTR, self.y2+Y_ENTR)

        # Increase velocity after 5 sec
        if time.time() - self.t0 >= 5:        
            
            self.speed_up()
            self.t0 = time.time()

    def speed_up(self):
        """
        Increase pipe speed
        """
        temp = self.vx
        self.vx *= 1.5
        if self.vx > MAX_PIPE_SPEED:
            self.vx = temp

    def reset(self):
        """
        Reset the obstacle to the starting position, and different h1
        """
        self.x = 800

        # Change obstacle heigth
        self.h1 = random.randint(30, HEIGHT - 1.5*self.gap)
        self.h2 = HEIGHT - self.gap - self.h1
        self.y2 = self.h1 + self.gap

        self.score_flag = True


def collision_detect(bird, pipes) -> bool:
    """
    Detect a collision between pipes and bird
    """
    collision = False

    for pipe in pipes:
        if pipe.x < bird.x < pipe.x + pipe.width:
            if not (pipe.h1 < bird.y < pipe.y2):
                collision=True
                
    
    return collision

def distance_score(bird: Bird, pipes: list[Obstacle], score) -> int:
    """
    Return the distance from the nearest pipe, increase score if needed
    """
    # Orizontal distance
    dist = np.array([pipe.x for pipe in pipes]) - np.array([bird.x for _ in range(PIPE_ON_SCREEN)])
    
    # Set distance to 1000 if bird passed the pipe
    for i in range(len(dist)):

        if dist[i] <= -pipes[i].width:      # First time that appens increase the score
            dist[i] = 1000
            if pipes[i].score_flag == True:
                score += 1
                pipes[i].score_flag = False        

        elif dist[i] <= 0:
            dist[i] = 1000

    nearest_pipe = np.argmin(dist)

    return min(dist), nearest_pipe, score

def entering_pipe(bird: Bird, pipes: list[Obstacle], index: int) -> bool:
    """
    Check if the bird is heading to the gap,
        parm: index -> index of the nearest pipe
    """
    # Output variable
    out = False

    # Save the nearest pipe
    near_pipe = pipes[index]

    # Get the nearest entering x coord and pipe x coord
    if near_pipe.x-X_ENTR < bird.x < near_pipe.x:

        # Get the y coord of the point of the line above and below the bird 
        y_top = -M*(bird.x-near_pipe.top_entr[0]) + near_pipe.top_entr[1]
        y_bot = M*(bird.x-near_pipe.bottom_entr[0]) + near_pipe.bottom_entr[1]
        
        if y_top < bird.y < y_bot:      # Bird in the entrance
            out = True

    return out