import pandas as pd
import numpy as np
import pygame
import random
import warnings
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── SETTINGS ──
WINDOW_W    = 1280
WINDOW_H    = 760
GRID_COLS   = 10
GRID_ROWS   = 10
CELL        = 56
MARGIN_L    = 20
MARGIN_T    = 70
PANEL_X     = MARGIN_L + GRID_COLS * CELL + 25
FPS         = 30

# ── COLOURS ──
BG           = (15, 20, 35)
SHELF_NORMAL = (45, 62, 95)
SHELF_TARGET = (30, 160, 255)
SHELF_DONE   = (50, 255, 100)
TEXT_COL     = (220, 230, 255)
PANEL_BG     = (22, 30, 50)
TITLE_COL    = (30, 160, 255)
AGENT_COLS   = [(255, 80, 80), (80, 200, 255),
                (255, 200, 50), (180, 110, 255)]

SCENARIO_COL = {
    'Normal':       (30, 160, 255),
    'Christmas':    (255, 70, 70),
    'Black Friday': (255, 180, 0),
}


class Robot:
    """A robot that navigates to target shelves
    using greedy shortest-path movement."""
    def __init__(self, rid, row, col):
        self.id = rid
        self.row = row
        self.col = col
        self.target = None
        self.carrying = False
        self.picks = 0
        self.steps = 0
        self.idle = 0

    def assign(self, target):
        self.target = target

    def step_toward(self):
        """Move one cell toward target
        (Manhattan/greedy)."""
        if self.target is None:
            self.idle += 1
            return False
        tr, tc = self.target
        moved = False
        if self.col < tc:
            self.col += 1; moved = True
        elif self.col > tc:
            self.col -= 1; moved = True
        elif self.row < tr:
            self.row += 1; moved = True
        elif self.row > tr:
            self.row -= 1; moved = True
        if moved:
            self.steps += 1
        # arrived?
        if (self.row, self.col) == self.target:
            self.picks += 1
            self.target = None
            return True  # completed a pick
        return False


def shelf_to_cell(shelf_id):
    s = shelf_id % (GRID_ROWS * GRID_COLS)
    return (s // GRID_COLS, s % GRID_COLS)


def draw(screen, fonts, robots, target_cells,
         done_cells, metrics, label, order_idx,
         total_orders, total_steps):
    font, small, title = fonts
    screen.fill(BG)

    # Title
    screen.blit(title.render(
        "Smart Warehouse Simulator", True,
        TITLE_COL), (MARGIN_L, 12))
    col = SCENARIO_COL.get(label, TEXT_COL)
    screen.blit(font.render(
        f"Scenario: {label}", True, col),
        (PANEL_X, 14))

    # Grid
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            sid = r * GRID_COLS + c
            x = MARGIN_L + c * CELL
            y = MARGIN_T + r * CELL
            rect = pygame.Rect(
                x, y, CELL - 3, CELL - 3)
            if (r, c) in done_cells:
                colour = SHELF_DONE
            elif (r, c) in target_cells:
                colour = SHELF_TARGET
            else:
                colour = SHELF_NORMAL
            pygame.draw.rect(
                screen, colour, rect,
                border_radius=6)
            screen.blit(small.render(
                str(sid), True, TEXT_COL),
                (x + 4, y + 4))

    # Robots
    for i, rob in enumerate(robots):
        x = MARGIN_L + rob.col * CELL + CELL // 2
        y = MARGIN_T + rob.row * CELL + CELL // 2
        ac = AGENT_COLS[i % len(AGENT_COLS)]
        # line to target
        if rob.target:
            tr, tc = rob.target
            tx = MARGIN_L + tc * CELL + CELL // 2
            ty = MARGIN_T + tr * CELL + CELL // 2
            pygame.draw.line(
                screen, ac, (x, y), (tx, ty), 2)
        pygame.draw.circle(
            screen, (0, 0, 0), (x + 2, y + 2), 15)
        pygame.draw.circle(screen, ac, (x, y), 15)
        screen.blit(small.render(
            f"R{rob.id+1}", True, (0, 0, 0)),
            (x - 9, y - 7))

    # Panel
    panel = pygame.Rect(
        PANEL_X - 10, 45,
        WINDOW_W - PANEL_X + 5, WINDOW_H - 55)
    pygame.draw.rect(
        screen, PANEL_BG, panel, border_radius=10)

    py = 58
    def line(txt, c=TEXT_COL, big=False, ind=0):
        nonlocal py
        f = font if big else small
        screen.blit(f.render(txt, True, c),
                    (PANEL_X + ind, py))
        py += 26 if big else 21

    line("LIVE METRICS", TITLE_COL, big=True)
    py += 4
    line(f"Order:  {order_idx} / {total_orders}")
    line(f"Steps:  {total_steps}")
    total_picks = sum(r.picks for r in robots)
    line(f"Total picks:  {total_picks}")
    eff = (total_picks / total_steps
           if total_steps > 0 else 0)
    line(f"Throughput:   {eff:.3f}")
    py += 10

    line("── ROBOTS ──", TITLE_COL)
    for i, rob in enumerate(robots):
        ac = AGENT_COLS[i % len(AGENT_COLS)]
        pygame.draw.circle(
            screen, ac, (PANEL_X + 12, py + 8), 7)
        line(f"   Robot {rob.id+1}:  "
             f"{rob.picks} picks, "
             f"{rob.steps} steps", ind=18)
    py += 12

    line("── EFFICIENCY ──", TITLE_COL)
    busy = sum(1 for r in robots
               if r.target is not None)
    line(f"   Active now:  {busy}/{len(robots)}")
    avg_picks = (total_picks / len(robots)
                 if robots else 0)
    line(f"   Avg picks/robot:  {avg_picks:.1f}")
    py += 14

    line("── SHELF KEY ──", TITLE_COL)
    for colour, lab in [
        (SHELF_NORMAL, "Idle shelf"),
        (SHELF_TARGET, "Target (robot en route)"),
        (SHELF_DONE,   "Just picked")]:
        pygame.draw.rect(
            screen, colour,
            (PANEL_X + 5, py, 14, 14),
            border_radius=3)
        screen.blit(small.render(
            lab, True, TEXT_COL),
            (PANEL_X + 26, py))
        py += 21

    pygame.display.flip()


def run_scenario(orders_df, label, screen, fonts,
                 clock, n_robots=4, max_orders=120):
    # Build order queue: each order -> list of
    # target shelves
    orders_df = orders_df.copy()
    orders_df['shelf'] = orders_df['aisle_id'] \
        % (GRID_ROWS * GRID_COLS)
    grouped = orders_df.groupby('order_id')[
        'shelf'].apply(list).tolist()[:max_orders]

    # Flatten into a task queue of shelves
    task_queue = []
    for order in grouped:
        for shelf in order:
            task_queue.append(shelf)

    # Robots start spread across grid
    robots = [Robot(i,
                    (i * 2) % GRID_ROWS,
                    (i * 3) % GRID_COLS)
              for i in range(n_robots)]

    total_steps = 0
    done_cells = {}      # cell -> frames left
    task_idx = 0
    orders_done = 0

    running = True
    while running and (task_idx < len(task_queue)
                       or any(r.target for r in robots)):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, True
            if event.type == pygame.KEYDOWN and \
               event.key == pygame.K_SPACE:
                return robots, False

        # Assign idle robots to next tasks
        for rob in robots:
            if rob.target is None and \
               task_idx < len(task_queue):
                shelf = task_queue[task_idx]
                rob.assign(shelf_to_cell(shelf))
                task_idx += 1

        # Step every robot
        target_cells = set()
        for rob in robots:
            if rob.target:
                target_cells.add(rob.target)
            completed = rob.step_toward()
            if completed:
                done_cells[(rob.row, rob.col)] = 12
                orders_done += 1
        total_steps += 1

        # Fade done cells
        for cell in list(done_cells):
            done_cells[cell] -= 1
            if done_cells[cell] <= 0:
                del done_cells[cell]

        draw(screen, fonts, robots, target_cells,
             set(done_cells.keys()),
             None, label, orders_done,
             len(task_queue), total_steps)
        clock.tick(FPS)

    return robots, False


def results_screen(screen, fonts, all_results):
    font, small, title = fonts
    screen.fill(BG)
    screen.blit(title.render(
        "FINAL RESULTS", True, TITLE_COL),
        (50, 40))
    screen.blit(font.render(
        "Press any key to exit", True, TEXT_COL),
        (50, 78))

    y = 140
    cols = ["Scenario", "Total Picks",
            "Total Steps", "Throughput",
            "Avg/Robot"]
    xs = [50, 300, 470, 650, 830]
    for i, c in enumerate(cols):
        screen.blit(font.render(c, True, TITLE_COL),
                    (xs[i], y))
    y += 36
    pygame.draw.line(screen, TITLE_COL,
                     (50, y), (980, y), 1)
    y += 12

    for label, robots in all_results:
        if robots is None:
            continue
        picks = sum(r.picks for r in robots)
        steps = max(r.steps for r in robots) \
            if robots else 0
        total_s = sum(r.steps for r in robots)
        thru = picks / total_s if total_s else 0
        avg = picks / len(robots) if robots else 0
        col = SCENARIO_COL.get(label, TEXT_COL)
        vals = [label, str(picks), str(total_s),
                f"{thru:.3f}", f"{avg:.1f}"]
        for i, v in enumerate(vals):
            screen.blit(font.render(v, True, col),
                        (xs[i], y))
        y += 32
    pygame.display.flip()

    wait = True
    while wait:
        for e in pygame.event.get():
            if e.type in (pygame.QUIT,
                          pygame.KEYDOWN):
                wait = False


def main():
    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_W, WINDOW_H))
    pygame.display.set_caption(
        "Smart Warehouse Simulator")
    clock = pygame.time.Clock()
    fonts = (
        pygame.font.SysFont("Arial", 17, bold=True),
        pygame.font.SysFont("Arial", 13),
        pygame.font.SysFont("Arial", 20, bold=True))

    print("\nSMART WAREHOUSE SIMULATOR")
    print("Robots navigate intelligently to shelves")
    print("\nChoose mode:")
    print("  1 = Compare Real vs Synthetic")
    print("  2 = Seasonal (Normal/Xmas/Black Fri)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        scenarios = [
            (pd.read_csv('data/normal_orders.csv'),
             "Normal"),
            (pd.read_csv('data/christmas_orders.csv'),
             "Christmas"),
            (pd.read_csv('data/blackfriday_orders.csv'),
             "Black Friday"),
        ]
    else:
        scenarios = [
            (pd.read_csv(
                'data/clean_orders_v2.csv').sample(
                n=20000, random_state=SEED),
             "Normal"),
            (pd.read_csv(
                'data/FINAL_synthetic_orders.csv'),
             "Christmas"),  # reuse colour slot
        ]

    all_results = []
    for df, label in scenarios:
        print(f"\nRunning: {label} "
              "(press SPACE to skip)")
        robots, quit_now = run_scenario(
            df, label, screen, fonts, clock)
        if quit_now:
            pygame.quit()
            return
        if robots:
            all_results.append((label, robots))
            picks = sum(r.picks for r in robots)
            print(f"  {label}: {picks} picks")

    results_screen(screen, fonts, all_results)
    pygame.quit()
    print("\nDone!")
    print(f"\n  {'Scenario':<14} {'Picks':>6} "
          f"{'Steps':>8} {'Throughput':>11}")
    for label, robots in all_results:
        if robots:
            picks = sum(r.picks for r in robots)
            steps = sum(r.steps for r in robots)
            thru = picks / steps if steps else 0
            print(f"  {label:<14} {picks:>6} "
                  f"{steps:>8} {thru:>11.4f}")


if __name__ == "__main__":
    main()