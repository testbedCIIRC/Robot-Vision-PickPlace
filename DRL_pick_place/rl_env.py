import pygame
import random
from enum import Enum
from collections import namedtuple
import time
import numpy as np

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
SPEED = 40

class GripperGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Gripper')
        self.clock = pygame.time.Clock()
        # init game state
        self.reset_env()

    def reset_env(self):
        self.place_packet()
        self.score = 0
        self.tracking_reward = 0
        self.gripper = None
        self.place_gripper()
        self.frame_count = 0


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
        
    def play_step(self,action):
        self.frame_count +=1
        # user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        self.move(action) # update the head
        self.packet.insert(0, self.head)
        reward = 0
        
        game_over = False
        max_frames = 300
        # if self.is_collision() or self.frame_count > max_frames:
        if self.is_collision():
            game_over = True
            reward = -20 + self.tracking_reward
            print('reward:', reward) 
            return reward, game_over, self.score

        if (self.packet[2].x == self.gripper.x or self.packet[3].x == self.gripper.x) and (self.gripper.y > int(self.h*0.7)):
            self.tracking_reward += 1
        # if self.gripper.y > self.h//2:
        #     self.tracking_reward += 1

        if self.packet[2] == self.gripper or self.packet[3] == self.gripper:
            self.score += 1
            reward = 20
            # time.sleep(0.5)
            # self.reset_env()
            self.place_packet()
            self.place_gripper()
        else:
            self.packet.pop()
        # update environment
        self.update_ui()
        self.clock.tick(SPEED)
        reward = reward + self.tracking_reward
        print('reward:', reward)
        return reward, game_over, self.score
    
    def is_collision(self,packet_head = None, gripper = None):
        if packet_head is None:
            packet_head = self.head

        if gripper is None:
            gripper = self.gripper

        if self.packet[0] == gripper or self.packet[1] == gripper or self.packet[4] == gripper or self.packet[5] == gripper:
            # time.sleep(0.5)
            return True
        # hits boundaries
        if packet_head.x > self.w - BLOCK_SIZE or packet_head.x < 0 or packet_head.y > self.h - BLOCK_SIZE or packet_head.y < 0:
            # time.sleep(0.5)
            return True
        if gripper.x > self.w - BLOCK_SIZE or gripper.x < 0 or gripper.y > self.h - BLOCK_SIZE or gripper.y < 0:
            # time.sleep(0.5)
            return True
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
        
    def move(self, action):
        
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

        x_h = self.head.x + BLOCK_SIZE
        y_h = self.head.y
        x_g = self.gripper.x 
        y_g = self.gripper.y
        if self.direction == Direction.RIGHT:
            x_g += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x_g -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y_g += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y_g -= BLOCK_SIZE

        self.gripper, self.head = Point(x_g, y_g), Point(x_h, y_h)
