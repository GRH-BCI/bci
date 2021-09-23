import time

import pygame
from collections import deque


class BCIGUI:
    def __init__(self, fullscreen=True) -> object:
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0) if fullscreen else (640, 480))
        self.background_colour = (100, 100, 100)
        self.message_font = pygame.font.SysFont(None, 30)
        self.message_colour = (255, 255, 255)
        self.message = None
        self.arrow_font = pygame.font.SysFont(None, 100)
        self.arrow_colour = (255, 255, 255)
        self.arrow_direction = None
        self.done = False
        self.recent_events = deque(maxlen=20)
        self.clock = clock = pygame.time.Clock()

    def loop(self):
        while not self.done:
            self.tick()
        pygame.quit()

    def tick(self):
        self._process_inputs()
        self._paint_all()
        pygame.display.flip()
        self.clock.tick(60)
        return not self.done

    def kill(self):
        self.done = True

    def set_text(self, text):
        self.message = self.message_font.render(text or '', True, self.message_colour)

    def set_arrow(self, direction):
        self.arrow_direction = direction

    def wait_for_click(self):
        self.recent_events.clear()
        while True:
            for event in self.recent_events:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return
            time.sleep(0.1)

    def _process_inputs(self):
        for event in pygame.event.get():
            self.recent_events.append(event)
            if event.type == pygame.QUIT:
                self.kill()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.kill()

    def _paint_all(self):
        self.screen.fill(self.background_colour)
        if self.message:
            self._paint(self.message, align='center')
        self._paint_arrow()

    def _paint(self, surface, xy=None, align='topleft'):
        if xy is not None:
            x, y = xy
        else:
            x, y = self._align_offset(align, self.screen.get_size())

        x_offset, y_offset = self._align_offset(align, surface.get_size())
        self.screen.blit(surface, (x-x_offset, y-y_offset))

    def _paint_arrow(self):
        arrow = self.arrow_font.render('>                     ', True, self.arrow_colour)
        if self.arrow_direction == 'left':
            arrow = pygame.transform.flip(arrow, True, False)
            self._paint(arrow, align='centerleft')
        elif self.arrow_direction == 'right':
            self._paint(arrow, align='centerright')
        elif self.arrow_direction == 'top':
            arrow = pygame.transform.rotate(arrow, 90)
            self._paint(arrow, align='topcenter')
        elif self.arrow_direction == 'bottom':
            arrow = pygame.transform.rotate(arrow, -90)
            self._paint(arrow, align='bottomcenter')

    def _align_offset(self, align, size):
        w, h = size
        return {
            'topleft': (0, 0),
            'bottomleft': (0, h),
            'centerleft': (0, h/2),
            'topright': (w, 0),
            'bottomright': (w, h),
            'centerright': (w, h/2),
            'topcenter': (w/2, 0),
            'bottomcenter': (w/2, h),
            'center': (w/2, h/2),
        }[align]
