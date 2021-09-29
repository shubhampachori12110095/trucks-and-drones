"""

"""
import sys
import numpy as np
import pygame
from pygame import Surface



class BaseVisualizer:

    def __init__(self, name, visual_params, temp_db):

        # Initialize pygame
        pygame.init()

        self.no_open_window = True

        # Define parameter:
        self.name      = name
        self.temp_db   = temp_db
        self.grid      = temp_db.grid
        [setattr(self, k, v) for k, v in visual_params.items()]

        # Define some colors
        self.color_dict = {
            'no_items_white': (240,240,240),
            'white': (255, 255, 255),
            'light-grey': (195, 195, 195),
            'grey': (128, 128, 128),
            'black': (0, 0, 0),
            'half_transp': (255, 255, 255, 125),
            'full_transp': (255, 255, 255, 0),
            'red': (165, 36, 36),
            'green':  (67, 149, 64),
            'blue': (81, 73, 186),
            'purple': (151, 69, 176),
            'light-blue': (65, 163, 212),
            'orange': (239, 179, 110),
            'yellow': (239, 203, 24),
        }
         
        #self.x_lenght = self.simulator.grid[0]
        #self.y_lenght = self.simulator.grid[1]

        # for resizing coordinates to grid window:
        self.x_mulipl = int(round(self.grid_surface_dim[0] / (self.grid[0])))
        self.y_mulipl = int(round(self.grid_surface_dim[1] / (self.grid[1])))

        self.axis_size = 20

        self.inner_grid_padding = self.marker_size * 2
        
        self.grid_surface_dim = [
            self.grid_surface_dim[0] + (self.marker_size * 4),
            self.grid_surface_dim[1] + (self.marker_size * 4)]
        
        # init grid surface:
        # the grid surface will display only markers without text
        # this surface will be used as image input for a conv net,
        # if the agent uses images as observations (states)
        self.grid_surface = Surface(self.grid_surface_dim, pygame.SRCALPHA)

        # init info grid surface:
        # this surface will put the grid coordinates to a marker
        # will also add more information, for example the cargo, stock, demand, if this is defined by the parameters
        # if this is defined by the parameter
        self.grid_info_surface = Surface(
            [self.grid_surface_dim[0]+ self.axis_size,
            self.grid_surface_dim[0] + self.axis_size]
            , pygame.SRCALPHA)

        # travel surface:
        self.travel_surface = Surface(self.grid_surface_dim, pygame.SRCALPHA)

        # init info surface:
        # creates an additional surface that will be displayed under the grid
        # this surface displays imformation about name, current episode and episode-step
        # can also diyplay more information via the add_info_dict, if this is defined by the parameters
        self.info_surface = Surface([self.grid_surface_dim[0], self.info_surface_height], pygame.SRCALPHA)

        # status surface:
        self.status_surface = Surface([
                500,
                self.grid_surface_dim[0] + 2 * self.grid_padding + self.info_surface_height + self.axis_size
            ], pygame.SRCALPHA
        )

        # init window
        self.window_width  = self.grid_surface_dim[0] + 2 * self.grid_padding + self.axis_size + 510
        self.window_height = self.grid_surface_dim[0] + 2 * self.grid_padding + self.info_surface_height + self.axis_size

        # init fonts:
        self.big_font    = pygame.font.SysFont('Arial', 12*2, bold=True)
        self.medium_font = pygame.font.SysFont('Arial', 10*2, bold=False)
        self.small_font  = pygame.font.SysFont('Arial', 6*2,  bold=False)

        # Block all events
        #pygame.event.set_blocked(None)


    def reset_surfaces(self):
        '''
        Erases old drawings by filling all surfaces with a fully transparent color.
        This will be called every step by visualize_step().
        '''
        self.grid_surface.fill(self.color_dict['full_transp'])
        self.grid_info_surface.fill(self.color_dict['full_transp'])
        self.travel_surface.fill(self.color_dict['full_transp'])
        self.info_surface.fill(self.color_dict['full_transp'])
        self.status_surface.fill(self.color_dict['full_transp'])
        self.screen.fill(self.color_dict['white'])

        for i in range(self.grid[0]+1):
            x_coord = self.small_font.render(str(i), True, self.color_dict['black'])
            width = int(round(x_coord.get_width()/2))
            self.grid_info_surface.blit(x_coord, (
                -width+self.axis_size + int(round(self.marker_size/2)) + int(round(self.marker_size*2)) + i * self.x_mulipl,
                int(round(self.axis_size/2))+(self.grid[1]+1)*self.x_mulipl
                )
            )

        for i in range(self.grid[1]+1):
            y_coord = self.small_font.render(str(self.grid[0]-i), True, self.color_dict['black'])
            height = int(round(y_coord.get_width()/2))
            self.grid_info_surface.blit(y_coord, (
                int(round(self.axis_size/2)),
                height-int(round(self.marker_size/2)) + int(round(self.marker_size*2)) + i*self.y_mulipl,
                )
            )



    def draw_circle_marker(self, coordinates, add_info=None, color='purple'):
        '''
        Draws a marker to the grid surface with a circle shape to specific location based on grid coordinates.
        Will also call draw_marker_info(), so additinal information can be drawn to the info grid surface.
        '''

        #resize grid coordinates to surface coordinates:
        surface_coordinates = [
            coordinates[0]*self.x_mulipl + int(round(self.marker_size*2)),
            (self.grid[1] - coordinates[1])*self.y_mulipl + int(round(self.marker_size*2))]

        # get relevant points:
        # A circle will be created from the middle of given coord,
        # so no further transformations are needed.
        x = surface_coordinates[0]
        y = surface_coordinates[1]

        # draw_circle
        pygame.draw.circle(self.grid_surface, self.color_dict[color], (x, y), int(round(self.marker_size / 2)))

        # info:
        #self.draw_marker_info(surface_coordinates, coordinates, add_info)


    def draw_rect_marker(self, coordinates, add_info=None, color='orange'):
        '''
        Draws a marker to the grid surface with a rectangle shape to specific location based on grid coordinates.
        Will also call draw_marker_info(), so additinal information can be drawn to the info grid surface.
        '''

        #resize grid coordinates to surface coordinates:
        surface_coordinates = [
            coordinates[0]*self.x_mulipl + int(round(self.marker_size*2)),
            (self.grid[1] - coordinates[1])*self.y_mulipl + int(round(self.marker_size*2))]
        
        # get relevant points:
        # A rectangle is drawn from the 'top' of given coordinates.
        # Note that the top coordinates are the lowest,
        # so we need to add a half of the marker size for x and y to be in the middle.
        x = surface_coordinates[0] - (self.marker_size/2)
        y = surface_coordinates[1] - (self.marker_size/2)

        # draw rectangle_
        pygame.draw.rect(self.grid_surface, self.color_dict[color], (x, y, self.marker_size, self.marker_size))

        # info:
        self.draw_marker_info(surface_coordinates, coordinates, add_info)


    def draw_triangle_down_marker(self, coordinates, add_info=None, color='light-blue'):
        '''
        Draws a marker to the grid surface with a traingle shape to specific location based on grid coordinates.
        Will also call draw_marker_info(), so additinal information can be drawn to the info grid surface.
        '''

        #resize grid coordinates to surface coordinates:
        surface_coordinates = [
            coordinates[0]*self.x_mulipl + int(round(self.marker_size*2)),
            (self.grid[1] - coordinates[1])*self.y_mulipl + int(round(self.marker_size*2))]

        # get relevant points:
        # A tringle is drawn by giving a polygon three points.
        # To be in the middle we apply the same logic of rectangles for y.
        # For one of the two x values we subtract the half of marker size,
        # for the other one we add a half.
        x_1 = surface_coordinates[0] + int(round(self.marker_size / 2))
        x_2 = surface_coordinates[0] - int(round(self.marker_size / 2))
        x_y = surface_coordinates[1] 
        y   = surface_coordinates[1] + int(round(self.marker_size / 2))
        y_x = surface_coordinates[0]

        # draw traingle:
        pygame.draw.polygon(self.grid_surface, self.color_dict[color], ([x_1, x_y], [y_x, y], [x_2, x_y]))

        # info:
        #self.draw_marker_info(surface_coordinates, coordinates, add_info)

    def draw_triangle_up_marker(self, coordinates, add_info=None, color='blue'):
        '''
        Draws a marker to the grid surface with a traingle shape to specific location based on grid coordinates.
        Will also call draw_marker_info(), so additinal information can be drawn to the info grid surface.
        '''

        #resize grid coordinates to surface coordinates:
        surface_coordinates = [
            coordinates[0]*self.x_mulipl + int(round(self.marker_size*2)),
            (self.grid[1] - coordinates[1])*self.y_mulipl + int(round(self.marker_size*2))]

        # get relevant points:
        # A tringle is drawn by giving a polygon three points.
        # To be in the middle we apply the same logic of rectangles for y.
        # For one of the two x values we subtract the half of marker size,
        # for the other one we add a half.
        x_1 = surface_coordinates[0] + int(round(self.marker_size / 2))
        x_2 = surface_coordinates[0] - int(round(self.marker_size / 2))
        x_y = surface_coordinates[1] 
        y   = surface_coordinates[1] - int(round(self.marker_size / 2))
        y_x = surface_coordinates[0]

        # draw traingle:
        pygame.draw.polygon(self.grid_surface, self.color_dict[color], ([x_1, x_y], [y_x, y], [x_2, x_y]))

        # info:
        #self.draw_marker_info(surface_coordinates, coordinates, add_info)

    def draw_marker_info(self, surface_coordinates, coordinates, add_info):
        '''
        Will be called by some draw marker function two draw some information to the info grid surface.
        '''

        # text to draw:
        text = '({},{},{})'.format(coordinates[0], coordinates[1], add_info)

        # create image from text:
        text = self.small_font.render(text, True, self.color_dict['grey'])
        
        # change surface coordinates, so no overlapping with marker occurs:
        x_text = surface_coordinates[0] + int(round(text.get_width() / 2))
        y_text = surface_coordinates[1] + int(round(self.marker_size / 2)) - 2
        
        # draw text to info grid surface:
        self.grid_info_surface.blit(text, (x_text, y_text))


    def draw_env_info(self, episode, step, add_info_dict=None):
        '''
        Draws some general information to the info surface.
        '''
        text_name  = self.big_font.render('Agent: '+self.name, True, self.color_dict['black'])
        text_episode = self.medium_font.render('Episode: '+str(episode), True, self.color_dict['black'])
        text_step  = self.medium_font.render('Step: '+str(step), True, self.color_dict['black'])

        big_height    = text_name.get_height()
        medium_height = text_episode.get_height()

        self.info_surface.blit(text_name,  (5, 5))
        self.info_surface.blit(text_episode, (5, 5 + big_height + 3))
        self.info_surface.blit(text_step,  (5, 5 + big_height + medium_height + 6))

        if isinstance(add_info_dict, dict):

            y = 5 + big_height + medium_height + 6
            for key in add_info_dict:
                text  = self.small_font.render(key+': '+str(add_info_dict[key]), True, self.color_dict['black'])
                y += self.small_height + 3
                self.info_surface.blit(text, (5, y))


    def draw_distance_traveled(self, episode, step, coordinates_list, color='grey'):
        '''
        ergänzen bei vehicle, temp_db update:
        arial travel: append start and end coordinates
        street travel: append start_coord, start_coord+[end_coord[0],0], end_coord
        '''
        surface_coordinates_list = [
            [elem[0]*self.x_mulipl + self.marker_size*2, (self.grid[1] - elem[1])*self.y_mulipl + self.marker_size*2] for elem in coordinates_list]
        if len(surface_coordinates_list) > 1:
            pygame.draw.lines(self.grid_surface, self.color_dict[color], False, points=surface_coordinates_list, width=1)


    def text_draw(self, i, text, fontsize='small', i_plus=1):
        if fontsize == 'small':
            text = self.small_font.render(text, True, self.color_dict['black'])
        elif fontsize == 'medium':
            text = self.medium_font.render(text, True, self.color_dict['black'])

        self.status_surface.blit(text, (5, 5 + i * 15))
        return i+i_plus

    def draw_status_dict(self):

        i = 0
        i = self.text_draw(i, 'Vehicles: ', 'medium', 2)
        keys = [
            'v_coord', 'v_dest', 'v_range', 'in_time_v_range', 'v_items', 'in_time_v_items',
            'v_cargo', 'in_time_v_cargo', 'loaded_v', 'in_time_loaded_v', 'v_free', 'v_stuck',
            'v_to_n'
        ]
        for key in keys:
            i = self.text_draw(i, (key+': '+str(np.round(self.temp_db.status_dict[key],2))))

        i += 1
        i = self.text_draw(i, 'Nodes: ', 'medium', 2)
        keys = [
            'n_coord', 'n_waiting', 'n_items', 'in_time_n_items',
        ]
        for key in keys:
            i = self.text_draw(i, (key+': '+str(np.round(self.temp_db.status_dict[key],2))))

        i += 1
        i = self.text_draw(i, 'Actions: ', 'medium', 2)
        for v_i in range(self.temp_db.num_vehicles):
            i = self.text_draw(i, str(self.temp_db.actions_list[v_i]))

        i += 1
        i = self.text_draw(i, 'Info: ', 'medium', 2)
        i = self.text_draw(i, 'cur_v_index: '+str(self.temp_db.cur_v_index))
        i = self.text_draw(i, 'cur_time_frame: '+str(self.temp_db.cur_time_frame))
        i = self.text_draw(i, 'time_till_fin: '+str(self.temp_db.time_till_fin))
        i = self.text_draw(i, 'total time ' + str(self.temp_db.total_time))
        i = self.text_draw(i, 'cargo loss ' + str(self.temp_db.signals_dict['cargo_loss']))

    def draw_marker_iter(self, marker_type):

        if marker_type == 'vehicle':
            iter_indices = self.temp_db.v_indices
            coord = self.temp_db.status_dict['v_coord']
            items = self.temp_db.status_dict['v_cargo']
            types = self.temp_db.constants_dict['v_type']
            symbols, colors = zip(*self.temp_db.vehicle_visuals)
            symbols = symbols
            colors = colors

        elif marker_type == 'node':
            iter_indices = self.temp_db.c_indices+self.temp_db.d_indices
            coord = self.temp_db.status_dict['n_coord']
            items = self.temp_db.status_dict['n_items']
            types = self.temp_db.constants_dict['n_type']
            symbols, colors = zip(*self.temp_db.node_visuals)
            symbols = list(symbols)
            colors = list(colors)

        for i in iter_indices:
            type_index = int(types[i])

            if items[i] == 0 and i in set(self.temp_db.c_indices):
                color = 'no_items_white'
            else:
                color = colors[type_index]

            if symbols[type_index] == 'circle':
                self.draw_circle_marker(coord[i], items[i], color=color)

            elif symbols[type_index] == 'triangle-up':
                self.draw_triangle_up_marker(coord[i], items[i], color=color)

            elif symbols[type_index] == 'triangle-down':
                self.draw_triangle_down_marker(coord[i], items[i], color=color)

            elif symbols[type_index] == 'rectangle':
                self.draw_rect_marker(coord[i], i, color=color)

            else:
                raise Exception(
                    "'symbol' was {}, but needs to be 'circle', 'triangle-up', 'triangle-down', 'rectangle'"
                )

    def visualize_step(self, episode, step, slow_down_pls=False):

        if self.no_open_window:
            self.screen = pygame.display.set_mode([self.window_width, self.window_height])
            pygame.display.set_caption(self.name)
            self.no_open_window = False

        self.reset_surfaces()

        self.draw_marker_iter('node')
        self.draw_marker_iter('vehicle')

        [self.draw_distance_traveled(episode, step, coordinates_list) for coordinates_list in self.temp_db.past_coord_not_transportable_v]
        [self.draw_distance_traveled(episode, step, coordinates_list) for coordinates_list in self.temp_db.past_coord_transportable_v]

        self.draw_env_info(episode, step)
        self.draw_status_dict()

        self.screen.blits((
            (self.travel_surface, (self.grid_padding + self.axis_size, self.grid_padding)),
            (self.grid_surface, (self.grid_padding+self.axis_size, self.grid_padding)),
            (self.grid_info_surface, (self.grid_padding - int(round(self.marker_size/2)), self.grid_padding)),
        ))
        
        self.screen.blit(self.info_surface,(self.grid_padding, self.grid_padding + self.grid_surface_dim[1]+self.axis_size))
        
        self.screen.blit(self.status_surface,(self.grid_padding+self.axis_size+self.grid_surface_dim[0], self.grid_padding))

        pygame.display.flip()

        if self.temp_db.debug_mode == True:
            self.wait_for_click()

        else:
            event_list = pygame.event.get()
            for event in event_list:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            for event in event_list:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.wait_for_click()

        if slow_down_pls:
            pygame.time.wait(250)

    def wait_for_click(self):
        event_happened = False
        while not event_happened:
            event = pygame.event.wait()
            if event.type == pygame.MOUSEBUTTONDOWN:
                # print(self.temp_db.status_dict)
                event_happened = True  #
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def convert_to_img_array(self):
        self.draw_marker_iter('node')
        self.draw_marker_iter('vehicle')
        return pygame.surfarray.array3d(self.grid_surface)

    def close(self):
        pygame.quit()
