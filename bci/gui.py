import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import time
import pygame
from collections import deque


class BCIGUI:
    """
    Provides a simple interface for SSVEP training built with `pygame`.

    The resulting GUI is a window (default fullscreen) that can show text centered on the screen, and an arrow
    that points either left, right, top or bottom.

    The intended use case is to run BCUIGUI.loop() one thread, then interact with the GUI asynchronously from other
    threads using `set_text()`, `set_arrow()`, `wait_for_click()` and `kill()`.
    """

    def __init__(self, fullscreen=True):
        """ Instantiates a BCIGUI object. """
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

    def loop(self):
        """ Repeatedly calls `self.tick()` until killed. """
        clock = pygame.time.Clock()
        while not self.done:
            self.tick()
            clock.tick(60)
        pygame.quit()

    def tick(self):
        """ Updates one frame of the GUI (processes inputs, paints text/arrows, etc.). """
        self._process_inputs()
        self._paint_all()
        pygame.display.flip()
        if self.done:
            pygame.quit()
        return not self.done

    def kill(self):
        """ Marks the GUI to be killed on next `tick()`. """
        self.done = True

    def set_text(self, text):
        """ Display text at the center of the screen, overwriting previous text if it was already there. """
        self.message = self.message_font.render(text or '', True, self.message_colour)

    def set_arrow(self, direction):
        """ Change which direction the arrow is pointing at ('left', 'right', 'top', or 'bottom'). """
        self.arrow_direction = direction

    def wait_for_click(self):
        """ Waits until the user clicks anywhere on the screen. Can be used to wait for user acknowledgement.
        Note that this doesn't `tick()` the GUI while waiting, so `loop()` must be called in another thread concurrent
        to `wait_for_click()`.
        """
        self.recent_events.clear()
        while not self.done:
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
