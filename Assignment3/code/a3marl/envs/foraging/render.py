import pygame
from pygame import Surface
from pygame.font import Font
from pygame.time import Clock

WINDOW_SIZE = (600, 600)
BG_COLOR = (245, 245, 245)
GRID_COLOR = (211, 211, 211)
AGENT_COLOR = (45, 45, 45)
CROP_COLOR = (220, 20, 60)
OBS_FILL_COLOR = (119, 158, 203, 100)
OBS_BORDER_COLOR = (70, 130, 180)
TEXT_DARK_COLOR = (50, 50, 50)
TEXT_LIGHT_COLOR = (255, 255, 255)


def _calculate_cell_size(grid_x_size: int, grid_y_size: int) -> int:
    return min(WINDOW_SIZE[0] // grid_y_size, WINDOW_SIZE[1] // grid_x_size)


class ForagingRenderer:
    def __init__(
        self,
        render_fps: int = 5,
        display_name: str = "Foraging Environment",
    ) -> None:
        self.render_fps: int = render_fps
        self.display_name: str = display_name

        self.pygame_viewer: Surface | None = None
        self.pygame_font: Font | None = None
        self.pygame_clock: Clock | None = None
        self.cell_size: float = 0
        self._is_initialized: bool = False

    def _initialize_pygame(self) -> None:
        if self._is_initialized:
            return
        pygame.init()
        self.pygame_viewer = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption(self.display_name)
        self.pygame_font = Font(None, 24)
        self.pygame_clock = Clock()
        self._is_initialized = True

    def draw_frame(
        self,
        xs: int,
        ys: int,
        agent_positions: list[tuple[int, int]],
        agent_levels: list[int],
        crop_positions: list[tuple[int, int]],
        crop_levels: list[int],
        crop_removed: list[bool],
        obs_radius: int,
    ) -> None:
        if not self._is_initialized:
            self._initialize_pygame()

        self.cell_size = _calculate_cell_size(xs, ys)

        canvas = Surface(WINDOW_SIZE)
        canvas.fill(BG_COLOR)

        for x in range(0, WINDOW_SIZE[0], self.cell_size):
            pygame.draw.line(canvas, GRID_COLOR, (x, 0), (x, WINDOW_SIZE[1]))
        for y in range(0, WINDOW_SIZE[1], self.cell_size):
            pygame.draw.line(canvas, GRID_COLOR, (0, y), (WINDOW_SIZE[0], y))

        for pos, level, removed in zip(crop_positions, crop_levels, crop_removed):
            if removed:
                continue
            x, y = pos
            center_x = int((y + 0.5) * self.cell_size)
            center_y = int((x + 0.5) * self.cell_size)
            radius = int(self.cell_size * 0.35)
            pygame.draw.circle(canvas, CROP_COLOR, (center_x, center_y), radius)
            if self.pygame_font:
                text_surf = self.pygame_font.render(str(level), True, TEXT_DARK_COLOR)
                text_rect = text_surf.get_rect(center=(center_x, center_y))
                canvas.blit(text_surf, text_rect)

        obs_overlay = Surface(WINDOW_SIZE, pygame.SRCALPHA)
        obs_rects = [
            pygame.Rect(
                (pos[1] - obs_radius) * self.cell_size,
                (pos[0] - obs_radius) * self.cell_size,
                (2 * obs_radius + 1) * self.cell_size,
                (2 * obs_radius + 1) * self.cell_size,
            )
            for pos in agent_positions
        ]
        for rect in obs_rects:
            pygame.draw.rect(obs_overlay, OBS_FILL_COLOR, rect)
        for rect in obs_rects:
            pygame.draw.rect(obs_overlay, OBS_BORDER_COLOR, rect, width=2)
        canvas.blit(obs_overlay, (0, 0))

        for pos, levels in zip(agent_positions, agent_levels):
            x, y = pos
            color = AGENT_COLOR

            square_size = int(self.cell_size * 0.7)
            offset = (self.cell_size - square_size) / 2
            rect_x = y * self.cell_size + offset
            rect_y = x * self.cell_size + offset

            agent_rect = pygame.Rect(rect_x, rect_y, square_size, square_size)
            pygame.draw.rect(canvas, color, agent_rect)

            if self.pygame_font:
                text_surf = self.pygame_font.render(str(levels), True, TEXT_LIGHT_COLOR)
                text_rect = text_surf.get_rect(center=agent_rect.center)
                canvas.blit(text_surf, text_rect)

        self.pygame_viewer.blit(canvas, (0, 0))
        pygame.display.flip()
        self.pygame_clock.tick(self.render_fps)

    def close(self) -> None:
        if self.pygame_viewer is not None:
            pygame.quit()
            self.pygame_viewer = None
            self._is_initialized = False
