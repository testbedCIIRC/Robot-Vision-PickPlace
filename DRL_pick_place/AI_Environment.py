import pygame
import random
from enum import Enum
from collections import namedtuple
import time

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')
WHITE = (255, 255, 255)
BROWN = (75,50,0)
RED = (255,0,0)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 5

class GripperGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Gripper')
        self.clock = pygame.time.Clock()
        # init game state
        self.direction = Direction.RIGHT
        self.place_packet()
        self.score = 0
        self.gripper = None
        self.place_gripper()

    def place_packet(self):
        self.direction = Direction.RIGHT
        self.head = Point(BLOCK_SIZE, (((self.h-BLOCK_SIZE)//BLOCK_SIZE ) *BLOCK_SIZE))
        self.packet = [self.head, 
                        Point(self.head.x-BLOCK_SIZE, self.head.y),
                        Point(self.head.x-(2*BLOCK_SIZE), self.head.y),
                        Point(self.head.x-(3*BLOCK_SIZE), self.head.y),
                        Point(self.head.x-(4*BLOCK_SIZE), self.head.y),
                        Point(self.head.x-(5*BLOCK_SIZE), self.head.y)]
    
    def place_gripper(self):
        x = (((self.w-BLOCK_SIZE)//BLOCK_SIZE) *BLOCK_SIZE ) - BLOCK_SIZE*20
        y = (((self.h-BLOCK_SIZE)//BLOCK_SIZE ) *BLOCK_SIZE) - BLOCK_SIZE*7
        self.gripper = Point(x, y)
        if self.gripper in self.packet:
            print('gripper')
            self.place_gripper()
        
    def play_step(self):
        # user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        self.move(self.direction) # update the head
        self.packet.insert(0, self.head)
    
        game_over = False
        if self.is_collision():
            game_over = True
            return game_over, self.score
        # place new gripper or just move
        if self.packet[2] == self.gripper or self.packet[3] == self.gripper:
            self.score += 1
            time.sleep(0.5)
            self.place_packet()
            self.place_gripper()
        else:
            self.packet.pop()
        # update environment
        self.update_ui()
        self.clock.tick(SPEED)
        return game_over, self.score
    
    def is_collision(self):
        if self.packet[0] == self.gripper or self.packet[1] == self.gripper or self.packet[4] == self.gripper or self.packet[5] == self.gripper:
            time.sleep(0.5)
            return True
        # hits boundaries
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            time.sleep(0.5)
            return True
        if self.gripper.x > self.w - BLOCK_SIZE or self.gripper.x < 0 or self.gripper.y > self.h - BLOCK_SIZE or self.gripper.y < 0:
            time.sleep(0.5)
            return True
        # hits itself
        # if self.head in self.packet[1:]:
        #     return True
        return False
        
    def update_ui(self):
        self.display.fill(BLACK)
        for pt in self.packet:
            pygame.draw.rect(self.display, BROWN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLACK, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display,  RED, pygame.Rect(self.gripper.x, self.gripper.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def move(self, direction):
        x_h = self.head.x + BLOCK_SIZE
        y_h = self.head.y
        x_g = self.gripper.x 
        y_g = self.gripper.y

        if direction == Direction.RIGHT:
            x_g += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x_g -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y_g += BLOCK_SIZE
        elif direction == Direction.UP:
            y_g -= BLOCK_SIZE

        self.gripper, self.head = Point(x_g, y_g), Point(x_h, y_h)

def start_game():
    game = GripperGame()
    while True:
        game_over, score = game.play_step()
        if game_over == True:
            break
    print('Final Score', score)
    start_game()

if __name__ == '__main__':
    start_game()   
    pygame.quit()