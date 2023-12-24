import time
import random


def initialize_background(width, height, tree_height):
    tree = initialize_tree(tree_height)
    background = [[" " for _ in range(width)] for _ in range(height)]

    # Center the tree horizontally and place it at the bottom vertically
    tree_start_x = (width - (2 * tree_height - 1)) // 2
    tree_start_y = height - tree_height - tree_height // 3  # Place tree at the bottom

    for i in range(len(tree)):
        for j in range(len(tree[i])):
            background[tree_start_y + i][tree_start_x + j] = tree[i][j]

    return background


def initialize_tree(height):
    green_color = "\033[92m"  # Green for leaves
    brown_color = "\033[33m"  # Brown for the trunk
    colors = ["\033[91m", "\033[93m", "\033[94m", "\033[95m"]  # Colors for bulbs
    reset_color = "\033[0m"

    tree = [[" " for _ in range(2 * height - 1)] for _ in range(height + height // 3)]

    for i in range(height):
        for j in range(height - i - 1, height + i):
            if j % 2 == 0:
                tree[i][j] = f"{green_color}*{reset_color}"
            else:
                if i % 2 == 1:
                    color_index = (i + j) % len(colors)
                    tree[i][j] = f"{colors[color_index]}o{reset_color}"
                else:
                    tree[i][j] = f"{green_color}*{reset_color}"

    trunk_width = height // 2
    trunk_height = height // 3
    trunk_start = height - trunk_width // 2 - 1
    for i in range(height, height + trunk_height):
        for j in range(trunk_start, trunk_start + trunk_width):
            tree[i][j] = f"{brown_color}|{reset_color}"

    return tree


def print_background(background):
    for row in background:
        print("".join(row))


def add_snow(background, snowflakes):
    width = len(background[0])
    for _ in range(snowflakes):
        x = random.randint(0, width - 1)
        if background[0][x] == " ":
            background[0][x] = "*"


def update_snow(background, tree_height):
    height = len(background)
    width = len(background[0])
    for i in range(height - 1, 0, -1):
        for j in range(width):
            if background[i][j] == " " and background[i - 1][j] == "*":
                background[i][j], background[i - 1][j] = "*", " "

    # Remove snow that hits the tree or goes beyond the bottom
    tree_start_x = (width - (2 * tree_height - 1)) // 2
    tree_end_x = tree_start_x + (2 * tree_height - 1)
    tree_start_y = (height - tree_height - tree_height // 3) // 2
    tree_end_y = tree_start_y + tree_height + tree_height // 3

    for i in range(tree_start_y, tree_end_y):
        for j in range(tree_start_x, tree_end_x):
            if background[i][j] != " " and background[i - 1][j] == "*":
                background[i - 1][j] = " "

    for i in range(width):
        if background[-1][i] == "*":
            background[-1][i] = " "


background_width = 90
background_height = 50
tree_height = 11
background = initialize_background(background_width, background_height, tree_height)

# Animation loop
while True:
    add_snow(background, 5)  # Add 5 snowflakes at the top
    update_snow(background, tree_height)  # Update the position of snowflakes
    print("\033c", end="")  # Clear the console
    print_background(background)  # Print the background with tree and snowflakes
    time.sleep(0.5)  # Half a second delay
