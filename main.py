import pygame as pg
import numpy as np
from numba import njit


SHADOW_DARKNESS = 0.5 #0: opaque, 1: invisible
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
MOUSE_SENSITIVITY = 50 # Default 50
SPRINT_SPEED = 2
WALK_SPEED = 1
CROUCH_SPEED = 0.5
horizontal_resolution = 350 #Game Resolution, Default 350 for 60fps
wall_texture_dim = 2048
Test = 0
def main():
    pg.init()
    screen = pg.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
    running = True
    clock = pg.time.Clock()
    pg.mouse.set_visible(False)
    pg.event.set_grab(1)
 
    half_vertical_resolution = int(horizontal_resolution * 0.375)
    scaling_factor = horizontal_resolution / 60
    step_size = 25
    player_x, player_y, player_rotation, map_heights, map_colors, exit_x, exit_y = generate_map(step_size)
    steps_count = player_x + player_y
    rotation_vertical = 0
    frame_buffer = np.random.uniform(0, 1, (horizontal_resolution, half_vertical_resolution * 2, 3))
    sky_texture = pg.image.load('Assets/Textures/me.png')
    sky_texture = pg.surfarray.array3d(pg.transform.smoothscale(sky_texture, (720, half_vertical_resolution * 4))) / 255
    floor_texture = pg.surfarray.array3d(pg.image.load('Assets/Textures/floor.png')) / 255
    wall_texture = pg.surfarray.array3d(pg.image.load('Assets/Textures/me.png')) / 255
    while running:
        ticks = pg.time.get_ticks() / 200
        elapsed_time = min(clock.tick() / 500, 0.3)
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                running = False
                
        frame_buffer = update_frame(player_x, player_y, player_rotation, frame_buffer, sky_texture, floor_texture,
                                    horizontal_resolution, half_vertical_resolution, scaling_factor, map_heights,
                                    step_size, wall_texture, map_colors, exit_x, exit_y, rotation_vertical)
        
        surface = pg.surfarray.make_surface(frame_buffer * 255)
        map_explored = np.zeros((step_size, step_size))
        surface = pg.transform.scale(surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
        screen.blit(surface, (0, 0))
        pg.display.update()
        fps = int(clock.get_fps())
        pg.display.set_caption(" FPS: " + str(fps))
        player_x, player_y, player_rotation, rotation_vertical = update_player_position(pg.key.get_pressed(), player_x,
                                                                                        player_y, player_rotation,
                                                                                        map_heights, elapsed_time,
                                                                                        rotation_vertical)

def update_player_position(pressed_keys, player_x, player_y, player_rotation, map_heights, elapsed_time, rotation_vertical):
    x, y, rotation, diagonal_movement = player_x, player_y, player_rotation, 0
    if pg.mouse.get_focused():
        mouse_position = pg.mouse.get_rel()
        rotation = rotation + np.clip((mouse_position[0]) / (16000 / MOUSE_SENSITIVITY), -0.2, .2)
        rotation_vertical = rotation_vertical + mouse_position[1] / (16000 / MOUSE_SENSITIVITY)
        rotation_vertical = np.clip(rotation_vertical, -1, 1)
    if pressed_keys[pg.K_LSHIFT]:
        player_speed = SPRINT_SPEED
    elif pressed_keys[pg.K_LCTRL]:
        player_speed = CROUCH_SPEED
    else:
        player_speed = WALK_SPEED
    if pressed_keys[pg.K_UP] or pressed_keys[ord('w')]:
        x, y, diagonal_movement = x + elapsed_time * player_speed*np.cos(rotation), y + elapsed_time * player_speed*np.sin(rotation), 1
    elif pressed_keys[pg.K_DOWN] or pressed_keys[ord('s')]:
        x, y, diagonal_movement = x - elapsed_time * player_speed*np.cos(rotation), y - elapsed_time * player_speed*np.sin(rotation), 1
    if pressed_keys[pg.K_LEFT] or pressed_keys[ord('a')]:
        elapsed_time = elapsed_time / (diagonal_movement + 1)
        x, y = x + elapsed_time * player_speed*np.sin(rotation), y - elapsed_time * player_speed*np.cos(rotation)
    elif pressed_keys[pg.K_RIGHT] or pressed_keys[ord('d')]:
        elapsed_time = elapsed_time / (diagonal_movement + 1)
        x, y = x - elapsed_time * player_speed*np.sin(rotation), y + elapsed_time * player_speed*np.cos(rotation)
    if pressed_keys[pg.K_e]:
        Test = 1
    elif pressed_keys[pg.K_q]:
        Test = 0
    player_x, player_y = check_collision(player_x, player_y, map_heights, x, y)

    return player_x, player_y, rotation, rotation_vertical

def generate_map(step_size):
    map_colors = np.random.uniform(0, 1, (step_size, step_size, 3)) 
    map_heights = np.random.choice([0, 0, 0, 0, 1, 2], (step_size, step_size))
    map_heights[0, :], map_heights[step_size - 1, :], map_heights[:, 0], map_heights[:, step_size - 1] = (1, 1, 1, 1)
    player_x, player_y = np.random.randint(1, step_size - 2) + 0.5, np.random.randint(1, step_size - 2) + 0.5
    player_rotation = np.pi / 4
    x, y = int(player_x), int(player_y)
    map_heights[x, y] = 0
    count = 0
    while True:
        test_x, test_y = (x, y)
        if np.random.uniform() > 0.5:
            test_x = test_x + np.random.choice([-1, 1])
        else:
            test_y = test_y + np.random.choice([-1, 1])
        if 0 < test_x < step_size - 1 and 0 < test_y < step_size - 1:
            if map_heights[test_x, test_y] == 0 or count > 5:
                count = 0
                x, y = (test_x, test_y)
                map_heights[x, y] = 0
                distance_to_exit = np.sqrt((x - player_x) ** 2 + (y - player_y) ** 2)
                if (distance_to_exit > step_size * 0.6 and np.random.uniform() > 0.999) or np.random.uniform() > 0.99999:
                    exit_x, exit_y = (x, y)
                    break
            else:
                count += 1
    return player_x, player_y, player_rotation, map_heights, map_colors, exit_x, exit_y
@njit()
def update_frame(player_x, player_y, player_rotation, frame_buffer, sky_texture, floor_texture, horizontal_resolution,
                 half_vertical_resolution, scaling_factor, map_heights, step_size, wall_texture, map_colors, exit_x,
                 exit_y, rotation_vertical):
    vertical_offset = -int(half_vertical_resolution * rotation_vertical)
    for i in range(horizontal_resolution):
        rotation_i = player_rotation + np.deg2rad(i / scaling_factor - 30)
        sin_rot, cos_rot, cos_rot2 = np.sin(rotation_i), np.cos(rotation_i), np.cos(np.deg2rad(i / scaling_factor - 30))
        frame_buffer[i][:] = sky_texture[int(np.rad2deg(rotation_i) * 2 % 718)][half_vertical_resolution - vertical_offset: 3 * half_vertical_resolution - vertical_offset]

        x, y = player_x, player_y
        while map_heights[int(x) % (step_size - 1)][int(y) % (step_size - 1)] == 0:
            x, y = x + 0.01 * cos_rot, y + 0.01 * sin_rot

        distance = np.sqrt((x - player_x) ** 2 + (y - player_y) ** 2)
        height = int(half_vertical_resolution / (distance * cos_rot2 + 0.001))

        xx = int(x % 1 * wall_texture_dim)        
        if 0.02 > x % 1 or x % 1 > 0.98:
            xx = int(y % 1 * wall_texture_dim)
        yy = np.linspace(0, 1, height * 2) * wall_texture_dim % wall_texture_dim

        shade = 0.3 + 0.7 * (height / half_vertical_resolution)
        if shade > 1:
            shade = 1
            
        ash = 0 
        if map_heights[int(x - 0.33) % (step_size - 1)][int(y - 0.33) % (step_size - 1)] != 0:
            ash = 1
            
        if map_heights[int(x - 0.01) % (step_size - 1)][int(y - 0.01) % (step_size - 1)] != 0:
            shade, ash = shade * 0.6, 0
            
        color = shade * map_colors[int(x) % (step_size - 1)][int(y) % (step_size - 1)]
        for k in range(height * 2):
            if half_vertical_resolution - height + k + vertical_offset >= 0 and half_vertical_resolution - height + k + vertical_offset < 2 * half_vertical_resolution:
                if ash and 1 - k / (2 * height) < 1 - xx / 99:
                    color, ash = 0.6 * color, 0
                frame_buffer[i][half_vertical_resolution - height + k + vertical_offset] = wall_texture[xx][int(yy[k])]
            if half_vertical_resolution + 3 * height - k + vertical_offset - 1 < half_vertical_resolution * 2: 
                frame_buffer[i][half_vertical_resolution + 3 * height - k + vertical_offset - 1] = wall_texture[xx][int(yy[k])]
                
        for j in range(half_vertical_resolution - height - vertical_offset): 
            distance = (half_vertical_resolution / (half_vertical_resolution - j - vertical_offset)) / cos_rot2
            x, y = player_x + cos_rot * distance, player_y + sin_rot * distance
            xx, yy = int(x * 3 % 1 * 99), int(y * 3 % 1 * 99)

            shade = min(0.2 + 0.8 / distance, 1)
            
            #Shadow Handler
            if map_heights[int(x - 0.33) % (step_size - 1)][int(y - 0.33) % (step_size - 1)] != 0:
                shade = shade * SHADOW_DARKNESS
            elif (map_heights[int(x - 0.33) % (step_size - 1)][int(y) % (step_size - 1)] and y % 1 > x % 1) or (map_heights[int(x) % (step_size - 1)][int(y - 0.33) % (step_size - 1)] and x % 1 > y % 1):
                shade = shade * SHADOW_DARKNESS
                
            frame_buffer[i][half_vertical_resolution * 2 - j - 1] = shade * (floor_texture[xx][yy] * 2 + frame_buffer[i][half_vertical_resolution * 2 - j - 1]) / 3
    return frame_buffer

def check_collision(player_x, player_y, map_heights, x, y): 
    if not (map_heights[int(x - 0.1)][int(y)] or map_heights[int(x + 0.1)][int(y)] or 
            map_heights[int(x)][int(y - 0.1)] or map_heights[int(x)][int(y + 0.1)]):
        player_x, player_y = x, y
        
    elif not (map_heights[int(player_x - 0.1)][int(y)] or map_heights[int(player_x + 0.1)][int(y)] or 
              map_heights[int(player_x)][int(y - 0.1)] or map_heights[int(player_x)][int(y + 0.1)]):
        player_y = y
        
    elif not (map_heights[int(x - 0.1)][int(player_y)] or map_heights[int(x + 0.1)][int(player_y)] or 
              map_heights[int(x)][int(player_y - 0.1)] or map_heights[int(x)][int(player_y + 0.1)]):
        player_x = x
        
    return player_x, player_y

if __name__ == '__main__':
    main()
    pg.mixer.quit()
    pg.quit()
