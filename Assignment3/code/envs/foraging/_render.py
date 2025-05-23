import pygame


class ForagingRenderer:
    """
    A class dedicated to handling the Pygame rendering for the Foraging environment.
    """

    def __init__(
        self,
        window_size: tuple[int, int] = (600, 600),
        render_fps: int = 5,
        display_name: str = "Foraging Environment",
        agent_color: tuple[int, int, int] = (0, 0, 255),
        crop_color: tuple[int, int, int] = (255, 0, 0),
        bg_color: tuple[int, int, int] = (200, 200, 200),
        grid_color: tuple[int, int, int] = (100, 100, 100),
        text_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.WINDOW_SIZE = window_size
        self.render_fps = render_fps
        self.display_name = display_name

        self.AGENT_COLOR = agent_color
        self.CROP_COLOR = crop_color
        self.BG_COLOR = bg_color
        self.GRID_COLOR = grid_color
        self.TEXT_COLOR = text_color

        self.pygame_viewer: pygame.Surface | None = None
        self.pygame_font: pygame.font.FontType | None = None
        self.pygame_clock: pygame.time.Clock | None = None
        self.cell_size: float = 0
        self._is_initialized: bool = False

    def _initialize_pygame(self) -> None:
        """Initializes Pygame, the display, font, and clock."""
        if self._is_initialized:
            return
        pygame.init()
        self.pygame_viewer = pygame.display.set_mode(self.WINDOW_SIZE)
        pygame.display.set_caption(self.display_name)
        self.pygame_font = pygame.font.Font(None, 24)  # Small font for levels
        self.pygame_clock = pygame.time.Clock()
        self._is_initialized = True

    def _calculate_cell_size(self, grid_x_size: int, grid_y_size: int) -> int:
        return min(
            self.WINDOW_SIZE[0] // grid_y_size, self.WINDOW_SIZE[1] // grid_x_size
        )

    def draw_frame(
        self,
        xs: int,
        ys: int,
        agent_positions: list[tuple[int, int]],
        agent_levels: list[int],
        crop_positions: list[tuple[int, int]],
        crop_levels: list[int],
        crop_removed: list[bool],
    ) -> None:
        """Draws a single frame of the environment."""
        if not self._is_initialized:
            self._initialize_pygame()

        self.cell_size = self._calculate_cell_size(xs, ys)

        canvas = pygame.Surface(self.WINDOW_SIZE)
        canvas.fill(self.BG_COLOR)

        # Draw grid lines
        for x in range(0, self.WINDOW_SIZE[0], self.cell_size):
            pygame.draw.line(canvas, self.GRID_COLOR, (x, 0), (x, self.WINDOW_SIZE[1]))
        for y in range(0, self.WINDOW_SIZE[1], self.cell_size):
            pygame.draw.line(canvas, self.GRID_COLOR, (0, y), (self.WINDOW_SIZE[0], y))

        for pos, level, removed in zip(crop_positions, crop_levels, crop_removed):
            if removed:
                continue
            x, y = pos
            rect = pygame.Rect(
                y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size
            )
            pygame.draw.rect(
                canvas,
                self.CROP_COLOR,
                rect.inflate(-self.cell_size * 0.4, -self.cell_size * 0.4),
            )
            if self.pygame_font:
                text_surf = self.pygame_font.render(str(level), True, self.TEXT_COLOR)
                text_rect = text_surf.get_rect(center=rect.center)
                canvas.blit(text_surf, text_rect)

        for pos, levels in zip(agent_positions, agent_levels):
            x, y = pos
            color = self.AGENT_COLOR

            center_x = int((y + 0.5) * self.cell_size)
            center_y = int((x + 0.5) * self.cell_size)
            radius = int(self.cell_size * 0.35)
            pygame.draw.circle(canvas, color, (center_x, center_y), radius)

            if self.pygame_font:
                text_surf = self.pygame_font.render(str(levels), True, self.TEXT_COLOR)
                text_rect = text_surf.get_rect(center=(center_x, center_y))
                canvas.blit(text_surf, text_rect)

        self.pygame_viewer.blit(canvas, (0, 0))
        pygame.display.flip()
        self.pygame_clock.tick(self.render_fps)

    def close(self):
        """Closes the Pygame display."""
        if self.pygame_viewer is not None:
            pygame.quit()
            self.pygame_viewer = None
            self._is_initialized = False
