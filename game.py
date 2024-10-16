import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1  # East
    DOWN = 2   # South
    LEFT = 3   # West
    UP = 4     # North

# just a regular tuple, but print(Point(1,1)) --> Point(x=1, y=1)
Point = namedtuple('Point', 'x, y')  

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)


class SnakeGameAI:

    def __init__(self, w=640, h=480, BLOCK_SIZE=20, SPEED=40, interactive_mode=False):
        self.w = w                      # game width
        self.h = h                      # game height
        self.BLOCK_SIZE = BLOCK_SIZE    # dist for one move
        self.SPEED = SPEED              # game speed
        # init display
        self.interactive_mode = interactive_mode
        if self.interactive_mode:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.quit = False
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        # snake is list of tuple of len 3
        #   head is middle of screen, and body in -x dir
        self.snake = [self.head,
                      Point(self.head.x-self.BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*self.BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-self.BLOCK_SIZE )//self.BLOCK_SIZE )*self.BLOCK_SIZE
        y = random.randint(0, (self.h-self.BLOCK_SIZE )//self.BLOCK_SIZE )*self.BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        if self.interactive_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit = True
                    # pygame.display.quit()
                    # quit()

                # from snake_game_human.py, not used here 
                """ ---- COMMENT OUT ---- 
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.direction = Direction.LEFT
                    elif event.key == pygame.K_RIGHT:
                        self.direction = Direction.RIGHT
                    elif event.key == pygame.K_UP:
                        self.direction = Direction.UP
                    elif event.key == pygame.K_DOWN:
                        self.direction = Direction.DOWN
                """
        
        # 2. move
        #   decode 3-state action vector, update the self.head tuple
        self._move(action) 
        #   add new self.head tuple to front of list (index=0)
        #   ex. [#, #, #] --> [self.head, #, #, #]
        self.snake.insert(0, self.head)
        
        # Reward: +10 if food, -10 if game_over, else 0
        # 3a. check if game over (collision OR its been 100*len(self.snake) frames)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        # 3b. check if game over (collision OR its been 100*len(self.snake) frames)
        if self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = 0
            return reward, game_over, self.score
        
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            # remove last num from list
            # ... since _move() inserts new head, pop() to move else grow longer
            self.snake.pop()  
        
        # 5. update ui and clock
        if self.interactive_mode:
            self._update_ui()
        self.clock.tick(self.SPEED)  # limit to run at self.SPEED FPS
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - self.BLOCK_SIZE or pt.x < 0 or pt.y > self.h - self.BLOCK_SIZE or pt.y < 0:
            return True
        # head hits body of snake
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, self.BLOCK_SIZE, self.BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, self.BLOCK_SIZE, self.BLOCK_SIZE))

        text = font.render("Score: " + str(self.score) , True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += self.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= self.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += self.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= self.BLOCK_SIZE

        self.head = Point(x, y)