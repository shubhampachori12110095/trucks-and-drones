"""

"""
import pygame


def reward_parameter(
        grid_surface_dim    = [400, 400],
        grid_padding        = 10,
        info_surface_height = 120,
        marker_size         = 6):
    return {
        'grid_surface_dim'   : grid_surface_dim,
        'grid_padding'       : grid_padding,
        'info_surface_height': info_surface_height,
        'marker_size'        : marker_size,
        }


class BaseVisualizor:

    def __init__(self, name, visual_param, simulator):

        # Initialize pygame
        pygame.init()c

        # Define parameter:
        self.name                = name
        self.simulator           = simulator
        [setattr(self, k, v) for k, v in visual_param.items()]

        # Define some colors
        self.black    = (0, 0, 0)
        self.grey     = (128, 128, 128)
        self.white    = (255, 255, 255)
        self.transp_h = (255, 255, 255, 125)
        self.transp_f = (255, 255, 255, 0)
        self.red      = (255, 0, 0)
        self.green    = (0, 255, 0)
        self.blue     = (0, 0, 255)
         
        #self.x_lenght = self.simulator.grid[0]
        #self.y_lenght = self.simulator.grid[1]

        # for resizing coordinates to grid window:
        self.x_mulipl = grid_surface_dim[0] // self.simulator.grid[0]
        self.y_mulipl = grid_surface_dim[1] // self.simulator.grid[1]

        # init grid surface:
        # the grid surface will display only markers without text
        # this surface will be used as image input for a conv net,
        # if the agent uses images as observations (states)
        self.grid_surface = Surface(grid_surface_dim)

        # init info grid surface:
        # this surface will put the grid coordinates to a marker
        # will also add more information, for example the cargo, stock, demand, if this is defined by the parameters
        # if this is defined by the parameter
        self.grid_info_surface = Surface(grid_surface_dim)

        # travel surface:
        self.travel_surface = Surface(grid_surface_dim)

        # init info surface:
        # creates an additional surface that will be displayed under the grid
        # this surface displays imformation about name, current episode and episode-step
        # can also diyplay more information via the add_info_dict, if this is defined by the parameters
        self.info_surface = Surface(grid_surface_dim[0], info_surface_height)

        # init window
        window_width  = grid_surface_dim[0] + 2 * grid_padding
        window_height = grid_surface_dim[0] + 2 * grid_padding + info_surface_height
        self.screen   = pygame.display.set_mode([window_width, window_height])

        # init fonts:
        self.big_font    = pygame.font.SysFont('Arial', 12, bold=True)
        self.medium_font = pygame.font.SysFont('Arial', 10, bold=False)
        self.small_font  = pygame.font.SysFont('Arial', 6,  bold=False)

        # Set title of screen
        pygame.display.set_caption(self.name)

        # Block all events
        pygame.event.set_blocked(None)


    def reset_surfaces(self):
        '''
        Erases old drawings by filling all surfaces with a fully transparent color.
        This will be called every step by visualize_step().
        '''
        self.grid_surface.fill(self.transp_f)
        self.grid_info_surface.fill(self.transp_f)
        self.travel_surface.fill(self.transp_f)
        self.info_surface.fill(self.transp_f)
        self.screen.fill(self.white)


    def draw_circle_marker(self, coordinates, color=(255, 0, 0), add_info=None):
        '''
        Draws a marker to the grid surface with a circle shape to specific location based on grid coordinates.
        Will also call draw_marker_info(), so additinal information can be drawn to the info grid surface.
        '''

        #resize grid coordinates to surface coordinates:
        surface_coordinates = [coordinates[0] * self.x_mulipl, coordinates[1] * self.y_mulipl]

        # get relevant points:
        # A circle will be created from the middle of given coord,
        # so no further transformations are needed.
        x = surface_coordinates[0]#-int(self.marker_size/2)
        y = surface_coordinates[1]#-int(self.marker_size/2)

        # draw_circle
        pygame.draw.circle(self.grid_surface, color, (x, y), self.marker_size // 2)

        # info:
        self.draw_marker_info(surface_coordinates, coordinates, add_info)


    def draw_rect_marker(self, coordinates, color=(0, 255, 0), add_info=None):
        '''
        Draws a marker to the grid surface with a rectangle shape to specific location based on grid coordinates.
        Will also call draw_marker_info(), so additinal information can be drawn to the info grid surface.
        '''

        #resize grid coordinates to surface coordinates:
        surface_coordinates = [coordinates[0] * self.x_mulipl, coordinates[1] * self.y_mulipl]
        
        # get relevant points:
        # A rectangle is drawn from the 'top' of given coordinates.
        # Note that the top coordinates are the lowest,
        # so we need to add a half of the marker size for x and y to be in the middle.
        x = surface_coordinates[0] + (self.marker_size // 2)
        y = surface_coordinates[1] + (self.marker_size // 2)

        # draw rectangle_
        pygame.draw.rect(self.grid_surface, color, (x, y, self.marker_size, self.marker_size))

        # info:
        self.draw_marker_info(coordinates, add_info)


    def draw_triangle_marker(self, coordinates, color=(0, 0, 255), add_info=None):
        '''
        Draws a marker to the grid surface with a traingle shape to specific location based on grid coordinates.
        Will also call draw_marker_info(), so additinal information can be drawn to the info grid surface.
        '''

        #resize grid coordinates to surface coordinates:
        surface_coordinates = [coordinates[0]*self.x_mulipl, coordinates[1]*self.y_mulipl]

        # get relevant points:
        # A tringle is drawn by giving a polygon three points.
        # To be in the middle we apply the same logic of rectangles for y.
        # For one of the two x values we subtract the half of marker size,
        # for the other one we add a half.
        x_1 = surface_coordinates[0] + (self.marker_size // 2)
        x_2 = surface_coordinates[0] - (self.marker_size // 2)
        y   = surface_coordinates[1] + (self.marker_size // 2)

        # draw traingle:
        pygame.draw.polygon(self.grid_surface, color, (x_1, x_2, y))

        # info:
        self.draw_marker_info(surface_coordinates, coordinates, add_info)


    def draw_marker_info(self, surface_coordinates, coordinates, add_info):
        '''
        Will be called by some draw marker function two draw some information to the info grid surface.
        '''

        # text to draw:
        text = '({},{},{})'.format(coordinates[0], coordinates[1], add_info)

        # create image from text:
        text = self.small_font.render(text, True, self.grey)
        
        # change surface coordinates, so no overlapping with marker occurs:
        x_text = surface_coordinates[0] + (text.get_width() // 2)
        y_text = surface_coordinates[1] + (self.marker_size // 2) - 2
        
        # draw text to info grid surface:
        self.grid_info_surface.blit(text, (x_text, y_text))


    def draw_env_info(self, episode, step, add_info_dict=None):
        '''
        Draws some general information to the info surface.
        '''
        text_name  = self.big_font.render('Agent: '+self.name, True, self.black)
        text_episode = self.medium_font.render('episode: '+str(episode), True, self.black)
        text_step  = self.medium_font.render('Step: '+str(step), True, self.black)

        big_height    = text.get_height()
        medium_height = text.get_height()

        self.info_surface.blit(text_name,  (5, 5))
        self.info_surface.blit(text_episode, (5, 5 + big_height + 3))
        self.info_surface.blit(text_step,  (5, 5 + big_height + medium_height + 6))

        if isinstance(add_info_dict, dict):

            y = 5 + big_height + medium_height + 6
            for key in add_info_dict:
                text  = self.small_font.render(key+': '+str(add_info_dict[key]), True, self.black)
                y += small_height + 3
                self.info_surface.blit(text, (5, y))


    def draw_distance_traveled(self, episode, step, coordinates_list, color=(0,0,0)):
        '''
        ergänzen bei vehicle, temp_db update:
        arial travel: append start and end coordinates
        street travel: append start_coord, start_coord+[end_coord[0],0], end_coord
        '''
        surface_coordinates_list = [(elem[0]*self.x_mulipl, elem[1]*self.y_mulipl) for elem in coordinates_list]
        pygame.draw.lines(self.travel_surface, color, False, surface_coordinates_list)


    def visualize_step(self):
        ###### NONE ERGÄNZEN!!!!!!!!!
        self.reset_surfaces()
        [self.draw_rect_marker(coordinates)            for coordinates in self.simulator.temp_db.cur_coord_nodes if coordinates not None]
        [self.draw_circle_marker(coordinates[-1])      for coordinates in self.simulator.temp_db.cur_coord_transportable_v if coordinates not None]
        [self.draw_triangle_marker(coordinates[-1])    for coordinates in self.simulator.temp_db.cur_coord_not_transportable_v if coordinates not None]

        [self.draw_distance_traveled(coordinates_list) for coordinates_list in self.simulator.temp_db.past_coord_not_transportable_v]
        [self.draw_distance_traveled(coordinates_list) for coordinates_list in self.simulator.temp_db.past_coord_transportable_v]

        self.draw_env_info(episode, step)

        self.screen.blit(self.grid_surface,      (self.grid_padding, self.grid_padding))
        self.screen.blit(self.grid_info_surface, (self.grid_padding, self.grid_padding))
        self.screen.blit(self.travel_surface,    (self.grid_padding, self.grid_padding))
        self.screen.blit(self.info_surface,      (self.grid_padding, self.grid_padding + self.grid_surface_dim[1]))

        pygame.display.flip()


    def close(self):
        pygame.quit()
