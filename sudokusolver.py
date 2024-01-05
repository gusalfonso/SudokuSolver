import numpy as np

def solver(game):
    for i in range(0, 9):
        for j in range(0, 9):
            if game[i][j] == 0:
                for n in range(1, 10):
                    if check(game, i, j, n):                         
                        game[i][j] = n
                        result = solver(game.copy())  # Make a copy to avoid modifying the original
                        if result is not None:
                            return result
                        game[i][j] = 0
                return None  # No solution found for this configuration
    return np.array(game) 

def check(game, x, y, n):
    # Check if n is already in the row or in the column:
    for i in range(0, 9):
        if game[x][i] == n or game[i][y] == n:
            return False

    # Check if n is already in any of the sub-square.
    x0 = (x//3)*3
    y0 = (y//3)*3

    for i in range(0, 3):
        for j in range(0, 3):
            if game[x0+i][y0+j] == n:
                return False
    return True
