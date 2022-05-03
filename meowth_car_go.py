import sys
import pygame
import pygame.camera
from car_control import *
from naive_bees import *
from random import randrange

pygame.init()
pygame.camera.init()
pygame.display.set_caption('meowth-car-remote-center')

screen = pygame.display.set_mode((640,480))
cam_list = pygame.camera.list_cameras()
cam = pygame.camera.Camera(cam_list[0],(640,480))
cam.start()
my_font = pygame.font.SysFont('Comic', 40)
running = True
while running:
    live_img = cam.get_image()
    live_img = pygame.transform.scale(live_img,(640,480))
    screen.blit(live_img,(0,0))
    pygame.display.update()
    pygame.time.Clock().tick(144)
    fps_overlay = my_font.render(
        str(pygame.time.Clock().get_fps())
                                 , 1, pygame.Color(0,0,0))
    screen.blit(fps_overlay, (0,0))
    # print("pre time: " + str(pygame.time.Clock().get_fps()))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            cam.stop()
            pygame.quit()            
            exit(0)
        elif event.type == pygame.KEYUP:
            gpio.cleanup()
        # keys control
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                print("finished")
                clean_up_gpio()
            elif event.key == pygame.K_LEFT:
                # print("==left==")
                left()
            elif event.key == pygame.K_RIGHT:
                # print("==right==")                
                right()
            elif event.key == pygame.K_UP:
                # print("==forward==") 
                forward()
            elif event.key == pygame.K_DOWN:
                # print("==backward==")
                backward()
            elif event.key == pygame.K_SPACE:
                print("==collect==")
                f_img = "/home/pi/Desktop/5642FinalProj/predict/bee/" + str(randrange(1000)) + ".jpg"
                img_data = pygame.transform.scale(live_img, (100,100))
                pygame.image.save(img_data, f_img)
                quick_pred(f_img)
                # bee_predict()
            else:
                clean_up_gpio()
