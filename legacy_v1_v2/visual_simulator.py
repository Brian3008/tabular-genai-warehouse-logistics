import pandas as pd
import numpy as np
import pygame
import gymnasium as gym
import rware
import time

# ── SETTINGS ──
WINDOW_W      = 1200
WINDOW_H      = 700
GRID_COLS     = 10
GRID_ROWS     = 10
CELL_SIZE     = 55
MARGIN_LEFT   = 20
MARGIN_TOP    = 60
PANEL_X       = MARGIN_LEFT + GRID_COLS * CELL_SIZE + 20
FPS           = 30

# ── COLOURS ──
BG_COLOUR       = (15,  20,  35)
GRID_COLOUR     = (40,  50,  70)
SHELF_COLOUR    = (50,  80, 120)
SHELF_ACTIVE    = (30, 160, 255)
AGENT_COLOURS   = [(255, 80,  80),
                   (80,  255, 120)]
TEXT_COLOUR     = (220, 230, 255)
PANEL_COLOUR    = (25,  32,  50)
SUCCESS_COLOUR  = (50,  255, 100)
TITLE_COLOUR    = (30,  160, 255)
BASELINE_COL    = (255, 160,  30)
CTGAN_COL       = (80,  220, 100)
REAL_COL        = (30,  160, 255)


def draw_warehouse(screen, font, small_font, shelves,
                   agent_positions, agent_targets,
                   metrics, current_label, step,
                   order_idx, total_orders, flash_shelves):

    screen.fill(BG_COLOUR)

    # ── TITLE ──
    title = font.render(
        "Warehouse Robot Simulation", True, TITLE_COLOUR)
    screen.blit(title, (MARGIN_LEFT, 15))

    mode_colours = {
        "Real Orders":        REAL_COL,
        "Shuffling Baseline": BASELINE_COL,
        "CTGAN Synthetic":    CTGAN_COL
    }
    col = mode_colours.get(current_label, TEXT_COLOUR)
    mode_txt = font.render(
        f"Mode: {current_label}", True, col)
    screen.blit(mode_txt, (PANEL_X, 15))

    # ── GRID ──
    for row in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x = MARGIN_LEFT + c * CELL_SIZE
            y = MARGIN_TOP  + row * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE-2, CELL_SIZE-2)

            shelf_id = row * GRID_COLS + c

            if shelf_id in flash_shelves:
                colour = SUCCESS_COLOUR
            elif shelf_id in shelves:
                colour = SHELF_ACTIVE
            else:
                colour = SHELF_COLOUR

            pygame.draw.rect(screen, colour, rect,
                             border_radius=6)

            # Shelf number
            lbl = small_font.render(
                str(shelf_id), True, TEXT_COLOUR)
            screen.blit(lbl, (x + 4, y + 4))

    # ── AGENTS ──
    for i, pos in enumerate(agent_positions):
        if pos is None:
            continue
        row, col_pos = pos
        x = MARGIN_LEFT + col_pos * CELL_SIZE + CELL_SIZE // 2
        y = MARGIN_TOP  + row     * CELL_SIZE + CELL_SIZE // 2

        # Shadow
        pygame.draw.circle(screen, (0, 0, 0),
                           (x+3, y+3), 16)
        # Agent body
        pygame.draw.circle(screen, AGENT_COLOURS[i],
                           (x, y), 16)
        # Agent label
        lbl = font.render(f"R{i+1}", True, (0, 0, 0))
        screen.blit(lbl, (x - 12, y - 10))

        # Target line
        if agent_targets[i] is not None:
            tr, tc = agent_targets[i]
            tx = MARGIN_LEFT + tc * CELL_SIZE + CELL_SIZE // 2
            ty = MARGIN_TOP  + tr * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.line(screen,
                             AGENT_COLOURS[i],
                             (x, y), (tx, ty), 2)
            pygame.draw.circle(screen,
                                AGENT_COLOURS[i],
                                (tx, ty), 6)

    # ── SIDE PANEL ──
    panel = pygame.Rect(PANEL_X - 10, 50,
                        WINDOW_W - PANEL_X, WINDOW_H - 60)
    pygame.draw.rect(screen, PANEL_COLOUR, panel,
                     border_radius=10)

    py = 70
    def draw_text(text, colour=TEXT_COLOUR,
                  big=False, offset=0):
        nonlocal py
        f = font if big else small_font
        s = f.render(text, True, colour)
        screen.blit(s, (PANEL_X + offset, py))
        py += 28 if big else 22

    draw_text("METRICS", TITLE_COLOUR, big=True)
    py += 5

    draw_text(f"Order:  {order_idx} / {total_orders}")
    draw_text(f"Step:   {step}")
    py += 8

    draw_text("── REAL ──", REAL_COL, big=False)
    draw_text(f"  Picks:  "
              f"{metrics['Real Orders']['picks']}")
    draw_text(f"  Reward: "
              f"{metrics['Real Orders']['reward']:.2f}")
    draw_text(f"  Steps:  "
              f"{metrics['Real Orders']['steps']}")
    py += 8

    draw_text("── BASELINE ──", BASELINE_COL)
    draw_text(f"  Picks:  "
              f"{metrics['Shuffling Baseline']['picks']}")
    draw_text(f"  Reward: "
              f"{metrics['Shuffling Baseline']['reward']:.2f}")
    draw_text(f"  Steps:  "
              f"{metrics['Shuffling Baseline']['steps']}")
    py += 8

    draw_text("── CTGAN ──", CTGAN_COL)
    draw_text(f"  Picks:  "
              f"{metrics['CTGAN Synthetic']['picks']}")
    draw_text(f"  Reward: "
              f"{metrics['CTGAN Synthetic']['reward']:.2f}")
    draw_text(f"  Steps:  "
              f"{metrics['CTGAN Synthetic']['steps']}")
    py += 15

    # Legend
    draw_text("── AGENTS ──", TEXT_COLOUR)
    for i, ac in enumerate(AGENT_COLOURS):
        pygame.draw.circle(screen, ac,
                           (PANEL_X + 15, py + 8), 8)
        lbl = small_font.render(
            f"Robot {i+1}", True, TEXT_COLOUR)
        screen.blit(lbl, (PANEL_X + 30, py))
        py += 22
    py += 10

    # Shelf legend
    draw_text("── SHELF STATUS ──", TEXT_COLOUR)
    for colour, label in [
        (SHELF_COLOUR,  "Normal shelf"),
        (SHELF_ACTIVE,  "Target shelf"),
        (SUCCESS_COLOUR,"Just picked!")
    ]:
        pygame.draw.rect(screen, colour,
                         (PANEL_X + 5, py, 16, 16),
                         border_radius=3)
        lbl = small_font.render(label, True, TEXT_COLOUR)
        screen.blit(lbl, (PANEL_X + 28, py))
        py += 22

    pygame.display.flip()


def run_visual_simulation(orders_df, label,
                           screen, font, small_font,
                           clock, all_metrics):

    env = gym.make("rware-tiny-2ag-v2")
    obs, info = env.reset()

    n_shelves    = GRID_ROWS * GRID_COLS
    orders_df    = orders_df.copy()
    orders_df['shelf_id'] = \
        orders_df['aisle_id'] % n_shelves

    unique_orders = orders_df.groupby('order_id').agg(
        shelves=('shelf_id', list),
        hour=('order_hour_of_day', 'first'),
        dow=('order_dow', 'first')
    ).reset_index().head(150)

    total_steps   = 0
    total_rewards = 0
    total_picks   = 0
    flash_shelves = set()
    flash_timer   = {}

    # Simulated agent positions on grid
    agent_positions = [(0, 0), (0, 1)]
    agent_targets   = [None, None]

    for order_idx, (_, order) in \
            enumerate(unique_orders.iterrows()):

        shelves_needed = order['shelves']

        for shelf_id in shelves_needed:
            target_row = (shelf_id % n_shelves) // GRID_COLS
            target_col = (shelf_id % n_shelves) %  GRID_COLS

            agent_idx = order_idx % 2
            agent_targets[agent_idx] = \
                (target_row, target_col)

            for step in range(150):
                # Handle quit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        pygame.quit()
                        return

                # Step environment
                actions = env.action_space.sample()
                obs, rewards, term, trunc, info = \
                    env.step(actions)

                if isinstance(rewards, (list, tuple)):
                    r = sum(rewards)
                else:
                    r = float(rewards)

                total_rewards += r
                total_steps   += 1

                if r > 0:
                    total_picks += 1
                    flash_shelves.add(shelf_id)
                    flash_timer[shelf_id] = 8

                # Move agent visually toward target
                ar, ac = agent_positions[agent_idx]
                tr, tc = target_row, target_col
                if ac < tc:
                    ac += 1
                elif ac > tc:
                    ac -= 1
                elif ar < tr:
                    ar += 1
                elif ar > tr:
                    ar -= 1
                agent_positions[agent_idx] = (ar, ac)

                # Update flash timers
                to_remove = []
                for sid in flash_shelves:
                    flash_timer[sid] -= 1
                    if flash_timer[sid] <= 0:
                        to_remove.append(sid)
                for sid in to_remove:
                    flash_shelves.discard(sid)
                    flash_timer.pop(sid, None)

                # Update metrics display
                all_metrics[label] = {
                    'picks':  total_picks,
                    'reward': total_rewards,
                    'steps':  total_steps
                }

                # Draw every frame
                active_shelves = set(
                    [s % n_shelves for s in shelves_needed])
                draw_warehouse(
                    screen, font, small_font,
                    active_shelves,
                    agent_positions,
                    agent_targets,
                    all_metrics,
                    label,
                    total_steps,
                    order_idx + 1,
                    len(unique_orders),
                    flash_shelves
                )
                clock.tick(FPS)

                if term or trunc:
                    obs, info = env.reset()
                    break

    env.close()
    return {
        'label':        label,
        'total_steps':  total_steps,
        'total_rewards': total_rewards,
        'total_picks':  total_picks
    }


# ── MAIN ──
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(
        "Tabular Gen-AI for Warehouse Logistics")
    clock = pygame.font.init()
    clock = pygame.time.Clock()

    font       = pygame.font.SysFont("Arial", 18, bold=True)
    small_font = pygame.font.SysFont("Arial", 14)

    print("Loading datasets...")
    real_df = pd.read_csv(
        'data/clean_orders.csv').sample(
        n=50000, random_state=42)
    baseline_df  = pd.read_csv('data/baseline_orders.csv')
    synthetic_df = pd.read_csv('data/synthetic_orders.csv')

    # Initialise metrics panel
    all_metrics = {
        'Real Orders':        {'picks': 0,
                               'reward': 0, 'steps': 0},
        'Shuffling Baseline': {'picks': 0,
                               'reward': 0, 'steps': 0},
        'CTGAN Synthetic':    {'picks': 0,
                               'reward': 0, 'steps': 0}
    }

    results = []

    for df, label in [
        (real_df,      "Real Orders"),
        (baseline_df,  "Shuffling Baseline"),
        (synthetic_df, "CTGAN Synthetic")
    ]:
        print(f"\nStarting: {label}")
        r = run_visual_simulation(
            df, label, screen, font,
            small_font, clock, all_metrics)
        if r:
            results.append(r)

    # Final summary screen
    screen.fill(BG_COLOUR)
    font_big = pygame.font.SysFont("Arial", 26, bold=True)
    y = 80
    title = font_big.render(
        "SIMULATION COMPLETE - FINAL RESULTS",
        True, TITLE_COLOUR)
    screen.blit(title, (50, y))
    y += 60

    headers = ["Condition", "Picks", "Rewards", "Steps"]
    xs      = [50, 350, 500, 650]
    for i, h in enumerate(headers):
        t = font.render(h, True, TITLE_COLOUR)
        screen.blit(t, (xs[i], y))
    y += 35

    colours = [REAL_COL, BASELINE_COL, CTGAN_COL]
    for i, r in enumerate(results):
        if r is None:
            continue
        vals = [r['label'],
                str(r['total_picks']),
                f"{r['total_rewards']:.2f}",
                str(r['total_steps'])]
        for j, v in enumerate(vals):
            t = font.render(v, True, colours[i])
            screen.blit(t, (xs[j], y))
        y += 30

    y += 30
    msg = small_font.render(
        "Close this window to exit.", True, TEXT_COLOUR)
    screen.blit(msg, (50, y))
    pygame.display.flip()

    # Wait for close
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False

    pygame.quit()
    print("\nVisual simulation complete!")


if __name__ == "__main__":
    main()