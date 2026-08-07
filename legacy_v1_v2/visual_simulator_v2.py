import pandas as pd
import numpy as np
import pygame
import gymnasium as gym
import rware
import time
import warnings
warnings.filterwarnings('ignore')

# ── SETTINGS ──
WINDOW_W    = 1280
WINDOW_H    = 750
GRID_COLS   = 10
GRID_ROWS   = 10
CELL_SIZE   = 55
MARGIN_LEFT = 20
MARGIN_TOP  = 70
PANEL_X     = MARGIN_LEFT + GRID_COLS * CELL_SIZE + 25
FPS         = 30

# ── COLOURS ──
BG           = (15,  20,  35)
GRID_NORMAL  = (45,  60,  90)
SHELF_NORMAL = (50,  80, 120)
SHELF_TARGET = (30, 160, 255)
SHELF_PICKED = (50, 255, 100)
AGENT_COLS   = [(255, 80,  80), (80, 200, 255),
                (255, 200, 50), (180, 80, 255)]
TEXT_COL     = (220, 230, 255)
PANEL_BG     = (20,  28,  48)
TITLE_COL    = (30,  160, 255)

SCENARIO_COLS = {
    'Real Orders':        (30,  160, 255),
    'Shuffling Baseline': (255, 160,  30),
    'CTGAN Synthetic':    (80,  220, 100),
    'Normal Day':         (30,  160, 255),
    'Christmas':          (255,  60,  60),
    'Black Friday':       (255, 180,   0),
}

SCENARIO_EMOJIS = {
    'Real Orders':        'REAL',
    'Shuffling Baseline': 'BASE',
    'CTGAN Synthetic':    'CTGAN',
    'Normal Day':         'NORMAL',
    'Christmas':          'XMAS',
    'Black Friday':       'BF',
}


def draw_warehouse(screen, font, small_font,
                   title_font,
                   target_shelves,
                   agent_positions,
                   agent_targets,
                   metrics, label,
                   step, order_idx,
                   total_orders,
                   flash_shelves,
                   mode):

    screen.fill(BG)

    # ── TITLE BAR ──
    col = SCENARIO_COLS.get(label, TEXT_COL)
    tag = SCENARIO_EMOJIS.get(label, label)

    title = title_font.render(
        "Tabular Gen-AI for Warehouse Logistics",
        True, TITLE_COL)
    screen.blit(title, (MARGIN_LEFT, 10))

    mode_surf = font.render(
        f"[{mode.upper()}]  {label}",
        True, col)
    screen.blit(mode_surf, (PANEL_X, 10))

    # ── GRID ──
    for row in range(GRID_ROWS):
        for c in range(GRID_COLS):
            shelf_id = row * GRID_COLS + c
            x = MARGIN_LEFT + c * CELL_SIZE
            y = MARGIN_TOP  + row * CELL_SIZE
            rect = pygame.Rect(
                x, y,
                CELL_SIZE-3, CELL_SIZE-3)

            if shelf_id in flash_shelves:
                colour = SHELF_PICKED
            elif shelf_id in target_shelves:
                colour = SHELF_TARGET
            else:
                colour = SHELF_NORMAL

            pygame.draw.rect(
                screen, colour, rect,
                border_radius=6)

            lbl = small_font.render(
                str(shelf_id), True, TEXT_COL)
            screen.blit(lbl, (x+4, y+4))

    # ── AGENTS ──
    n_agents = len(agent_positions)
    for i, pos in enumerate(agent_positions):
        if pos is None:
            continue
        r, c = pos
        x = MARGIN_LEFT + c*CELL_SIZE + CELL_SIZE//2
        y = MARGIN_TOP  + r*CELL_SIZE + CELL_SIZE//2

        pygame.draw.circle(
            screen, (0,0,0), (x+3, y+3), 14)
        pygame.draw.circle(
            screen, AGENT_COLS[i % len(AGENT_COLS)],
            (x, y), 14)
        lbl = small_font.render(
            f"R{i+1}", True, (0,0,0))
        screen.blit(lbl, (x-10, y-8))

        if agent_targets[i] is not None:
            tr, tc = agent_targets[i]
            tx = MARGIN_LEFT+tc*CELL_SIZE+CELL_SIZE//2
            ty = MARGIN_TOP +tr*CELL_SIZE+CELL_SIZE//2
            pygame.draw.line(
                screen,
                AGENT_COLS[i % len(AGENT_COLS)],
                (x,y), (tx,ty), 1)

    # ── SIDE PANEL ──
    panel = pygame.Rect(
        PANEL_X-10, 45,
        WINDOW_W-PANEL_X+5, WINDOW_H-55)
    pygame.draw.rect(
        screen, PANEL_BG, panel,
        border_radius=10)

    py = 60

    def txt(text, colour=TEXT_COL,
            big=False, indent=0):
        nonlocal py
        f = font if big else small_font
        s = f.render(text, True, colour)
        screen.blit(s, (PANEL_X+indent, py))
        py += 26 if big else 20

    txt("LIVE METRICS", TITLE_COL, big=True)
    py += 4
    txt(f"Order:  {order_idx} / {total_orders}")
    txt(f"Step:   {step}")
    py += 8

    for sc_label, sc_metrics in metrics.items():
        sc_col = SCENARIO_COLS.get(
            sc_label, TEXT_COL)
        tag = SCENARIO_EMOJIS.get(
            sc_label, sc_label)
        txt(f"── {tag} ──", sc_col)
        txt(f"  Picks:   "
            f"{sc_metrics['picks']}")
        txt(f"  Reward:  "
            f"{sc_metrics['reward']:.1f}")
        txt(f"  Steps:   "
            f"{sc_metrics['steps']}")
        py += 6

    py += 10
    txt("── AGENTS ──", TEXT_COL)
    for i in range(n_agents):
        pygame.draw.circle(
            screen,
            AGENT_COLS[i % len(AGENT_COLS)],
            (PANEL_X+14, py+8), 8)
        txt(f"  Robot {i+1}", TEXT_COL,
            indent=20)

    py += 8
    txt("── SHELF STATUS ──", TEXT_COL)
    for colour, label_txt in [
        (SHELF_NORMAL, "Normal shelf"),
        (SHELF_TARGET, "Target shelf"),
        (SHELF_PICKED, "Just picked!")
    ]:
        pygame.draw.rect(
            screen, colour,
            (PANEL_X+5, py, 14, 14),
            border_radius=3)
        lbl = small_font.render(
            label_txt, True, TEXT_COL)
        screen.blit(lbl, (PANEL_X+26, py))
        py += 20

    pygame.display.flip()


def run_visual_sim(orders_df, label,
                   screen, font,
                   small_font, title_font,
                   clock, all_metrics,
                   n_agents=4, mode="comparison"):

    env = gym.make(
        f"rware-small-{n_agents}ag-v2")
    obs, info = env.reset()

    n_shelves = GRID_ROWS * GRID_COLS
    orders_df = orders_df.copy()
    orders_df['shelf_id'] = \
        orders_df['aisle_id'] % n_shelves

    unique_orders = orders_df.groupby(
        'order_id').agg(
        shelves=('shelf_id', list),
        hour=('order_hour_of_day', 'first'),
        dow=('order_dow', 'first')
    ).reset_index().head(150)

    total_steps   = 0
    total_rewards = 0
    total_picks   = 0
    flash_shelves = set()
    flash_timer   = {}

    agent_positions = [
        (i // GRID_COLS, i % GRID_COLS)
        for i in range(n_agents)]
    agent_targets = [None] * n_agents

    for order_idx, (_, order) in enumerate(
            unique_orders.iterrows()):

        shelves_needed = order['shelves']

        for shelf_id in shelves_needed:
            target_row = \
                (shelf_id % n_shelves) // GRID_COLS
            target_col = \
                (shelf_id % n_shelves) %  GRID_COLS

            agent_idx = order_idx % n_agents
            agent_targets[agent_idx] = \
                (target_row, target_col)

            for step in range(120):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        pygame.quit()
                        return None
                    if event.type == \
                            pygame.KEYDOWN:
                        if event.key == \
                                pygame.K_SPACE:
                            print(
                                "  Skipping to "
                                "next scenario...")
                            env.close()
                            return {
                                'label': label,
                                'total_steps':
                                    total_steps,
                                'total_rewards':
                                    total_rewards,
                                'total_picks':
                                    total_picks
                            }

                actions = env.action_space.sample()
                obs, rewards, term, trunc, info\
                    = env.step(actions)

                r = sum(rewards) \
                    if isinstance(
                        rewards, (list, tuple))\
                    else float(rewards)

                total_rewards += r
                total_steps   += 1
                if r > 0:
                    total_picks += 1
                    flash_shelves.add(shelf_id)
                    flash_timer[shelf_id] = 10

                # Move agent toward target
                ar, ac = agent_positions[agent_idx]
                tr2, tc2 = target_row, target_col
                if   ac < tc2: ac += 1
                elif ac > tc2: ac -= 1
                elif ar < tr2: ar += 1
                elif ar > tr2: ar -= 1
                agent_positions[agent_idx] = (ar, ac)

                # Flash timer
                for sid in list(flash_shelves):
                    flash_timer[sid] -= 1
                    if flash_timer[sid] <= 0:
                        flash_shelves.discard(sid)
                        flash_timer.pop(sid, None)

                all_metrics[label] = {
                    'picks':  total_picks,
                    'reward': total_rewards,
                    'steps':  total_steps
                }

                target_set = set(
                    [s % n_shelves
                     for s in shelves_needed])

                draw_warehouse(
                    screen, font, small_font,
                    title_font,
                    target_set,
                    agent_positions,
                    agent_targets,
                    all_metrics, label,
                    total_steps,
                    order_idx+1,
                    len(unique_orders),
                    flash_shelves,
                    mode)

                clock.tick(FPS)

                if term or trunc:
                    obs, info = env.reset()
                    break

    env.close()
    return {
        'label':         label,
        'total_steps':   total_steps,
        'total_rewards': total_rewards,
        'total_picks':   total_picks
    }


def show_results_screen(screen, font,
                         small_font,
                         title_font,
                         results, mode):
    screen.fill(BG)

    title = title_font.render(
        f"RESULTS — {mode.upper()} MODE",
        True, TITLE_COL)
    screen.blit(title, (50, 40))

    sub = font.render(
        "Press any key to exit",
        True, TEXT_COL)
    screen.blit(sub, (50, 80))

    y = 140
    headers = ["Scenario", "Picks",
               "Rewards", "Steps"]
    xs = [50, 380, 520, 660]
    for i, h in enumerate(headers):
        t = font.render(h, True, TITLE_COL)
        screen.blit(t, (xs[i], y))
    y += 35

    pygame.draw.line(
        screen, TITLE_COL,
        (50, y), (800, y), 1)
    y += 10

    for r in results:
        if r is None:
            continue
        col = SCENARIO_COLS.get(
            r['label'], TEXT_COL)
        vals = [
            r['label'],
            str(r['total_picks']),
            f"{r['total_rewards']:.1f}",
            str(r['total_steps'])
        ]
        for j, v in enumerate(vals):
            t = font.render(v, True, col)
            screen.blit(t, (xs[j], y))
        y += 32

    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               event.type == pygame.KEYDOWN:
                waiting = False


def main():
    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_W, WINDOW_H))
    pygame.display.set_caption(
        "Tabular Gen-AI for Warehouse Logistics")

    clock      = pygame.time.Clock()
    title_font = pygame.font.SysFont(
        "Arial", 20, bold=True)
    font       = pygame.font.SysFont(
        "Arial", 17, bold=True)
    small_font = pygame.font.SysFont(
        "Arial", 13)

    print("\n" + "="*60)
    print("WAREHOUSE GEN-AI VISUAL SIMULATOR v2")
    print("="*60)
    print("\nChoose simulation mode:")
    print("  1 → Comparison mode")
    print("      Real vs Baseline vs CTGAN")
    print("  2 → Seasonal mode")
    print("      Normal vs Christmas vs Black Friday")
    print("  3 → Full mode (all scenarios)")
    print("\nPress SPACE during simulation")
    print("to skip to next scenario")

    choice = input("\nEnter 1, 2 or 3: ").strip()

    if choice == "1":
        mode = "comparison"
        scenarios = [
            (pd.read_csv(
                'data/clean_orders.csv').sample(
                n=50000, random_state=42),
             "Real Orders"),
            (pd.read_csv(
                'data/baseline_orders.csv'),
             "Shuffling Baseline"),
            (pd.read_csv(
                'data/FINAL_synthetic_orders.csv'),
             "CTGAN Synthetic"),
        ]
    elif choice == "2":
        mode = "seasonal"
        scenarios = [
            (pd.read_csv(
                'data/normal_orders.csv'),
             "Normal Day"),
            (pd.read_csv(
                'data/christmas_orders.csv'),
             "Christmas"),
            (pd.read_csv(
                'data/blackfriday_orders.csv'),
             "Black Friday"),
        ]
    else:
        mode = "full"
        scenarios = [
            (pd.read_csv(
                'data/clean_orders.csv').sample(
                n=50000, random_state=42),
             "Real Orders"),
            (pd.read_csv(
                'data/FINAL_synthetic_orders.csv'),
             "CTGAN Synthetic"),
            (pd.read_csv(
                'data/normal_orders.csv'),
             "Normal Day"),
            (pd.read_csv(
                'data/christmas_orders.csv'),
             "Christmas"),
            (pd.read_csv(
                'data/blackfriday_orders.csv'),
             "Black Friday"),
        ]

    # Initialise metrics
    all_metrics = {
        label: {'picks': 0,
                'reward': 0,
                'steps': 0}
        for _, label in scenarios
    }

    results = []

    for df, label in scenarios:
        print(f"\nStarting: {label}")
        print("(Press SPACE to skip scenario)")
        r = run_visual_sim(
            df, label, screen,
            font, small_font, title_font,
            clock, all_metrics,
            n_agents=4, mode=mode)
        if r:
            results.append(r)
            print(f"  Completed: {label}")
            print(f"  Picks: {r['total_picks']} | "
                  f"Steps: {r['total_steps']}")

    # Show final results screen
    if results:
        show_results_screen(
            screen, font, small_font,
            title_font, results, mode)

    pygame.quit()
    print("\nSimulation complete!")
    print("\nFinal results:")
    for r in results:
        if r:
            print(f"  {r['label']:<20} "
                  f"Picks: {r['total_picks']:>3} | "
                  f"Steps: {r['total_steps']:>6}")


if __name__ == "__main__":
    main()