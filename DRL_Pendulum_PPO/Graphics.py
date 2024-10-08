import pygame
from math import sin, cos
from parameters import MAX_ANGLE, SMALL_ANGLE
from time import time

class GUI():

    def __init__(self):
        

        # Colors definition
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)

        # Windows parameters
        self.width, self.height = 800, 600

        # Pendulum parameters
        self.center = (self.width//2, self.height//2)          # used // to get an int after division
        self.lenght = 200

        self.max_reward = -100
        self.max_len = 0
        


    def GUI_init(self, t0):
        # Call to create a screen
        pygame.init()
        
        # Font
        self.font = pygame.font.SysFont('Arial', 32)
        pygame.font.init()
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Inversed pendulum')
        
        # Store initial time
        self.t0 = t0

    def GUI_quit(self) -> bool:
        # Close window to terminate the execution
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True  

    def draw(self, angle : float, reset : int, reward : int, FPS = 60) -> None:

        # End point coordinate
        x = int(self.center[0] + self.lenght*sin(angle))
        y = int(self.center[1] - self.lenght*cos(angle)) 

        # Good range
        x3, y3 = int(self.center[0] + self.lenght*sin(SMALL_ANGLE)), int(self.center[0] - self.lenght*cos(SMALL_ANGLE)*2) 
        x4, y4 = int(self.center[0] + self.lenght*sin(-SMALL_ANGLE)), int(self.center[0] - self.lenght*cos(-SMALL_ANGLE)*2) 


        # Graphics drawing
        self.screen.fill(self.WHITE)
        pygame.draw.line(self.screen, self.BLACK, self.center, (x,y), 4)

        pygame.draw.line(self.screen, self.RED, (0,self.height//2), (self.width, self.height//2),1)

        pygame.draw.line(self.screen, self.GREEN, self.center, (x3,y3),1)
        pygame.draw.line(self.screen, self.GREEN, self.center, (x4,y4),1)

        pygame.draw.circle(self.screen, self.BLUE, self.center, 10)

        text = self.font.render(f"N Reset: {reset}, Time: {time()-self.t0:.1f} s , Rew: {reward}",True, self.BLACK)
        
        self.screen.blit(text, (20, self.height-50))

        pygame.display.flip()

        pygame.time.Clock().tick(FPS)


        
        

