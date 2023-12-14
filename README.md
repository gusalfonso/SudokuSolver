The Art of Recursion in Python: A Sudoku Solver Journey

Recursion is a powerful programming concept that allows a function to call itself, providing an elegant solution to problems that exhibit repetitive and self-similar structures. In this article, we'll explore the beauty of recursion in Python through the lens of a Sudoku solver.

The Sudoku Challenge
Sudoku, a popular numerical puzzle, serves as an excellent playground for exploring recursion. The goal is to fill a 9x9 grid with digits from 1 to 9, adhering to specific rules: each row, each column, and each 3x3 subgrid must contain all the digits without repetition.

The Recursive Sudoku Solver
Let's dissect a Python function that embodies the essence of recursion in solving Sudoku puzzles:

def solver(game):
    # For each row
    for i in range(0, 9):
        # For each column
        for j in range(0, 9):
            if game[i][j] == 0:
                for n in range(1, 10):
                    if check(game, i, j, n):
                        game[i][j] = n
                        solver(game)
                        game[i][j] = 0
                return
    print(np.matrix(game))

The Recursive Dance
Exploring Possibilities:

The outer loop iterates over each row.
The inner loop iterates over each column.
If a cell is empty (contains 0), the function enters a recursive exploration.
Recursive Exploration:

For each empty cell, the function tries numbers from 1 to 9.
The check function verifies if the number is valid in terms of Sudoku rules.
If valid, the number is saved in the cell, and the function recursively explores the next empty cell.
Backtracking:

If the recursive exploration reaches a dead-end (no valid number for the current cell), the function backtracks.
The current cell is reset to 0, allowing the algorithm to try different numbers in the previous cell.
The Validator Function
The check function is crucial in determining the validity of a number placement:

def check(game, x, y, n):
    # Check if n is already in the row or column
    for i in range(0, 9):
        if game[x][i] == n or game[i][y] == n:
            return False

    # Check if n is already in any of the sub-square
    x0 = (x // 3) * 3
    y0 = (y // 3) * 3

    for i in range(0, 3):
        for j in range(0, 3):
            if game[x0 + i][y0 + j] == n:
                return False
    return True

Conclusion: Unraveling the Recursion Magic
Recursion, as demonstrated in this Sudoku solver example, unveils its magic by elegantly handling complex problems through a divide-and-conquer approach. The solver explores countless possibilities, gracefully backtracking when needed, until it unveils the solution to the Sudoku puzzle. The recursive dance in Python showcases the versatility and power of this programming paradigm, making it an invaluable tool in a programmer's toolkit.




