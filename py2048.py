import numpy as np

class GameBoard:
    """represents a gameboard for the game 2048."""
    def __init__(self, height=4, width=4):
        self.height = height
        self.width = width
        self.board = np.zeros(shape=(height, width))
        self.left = 0
        self.up = 1
        self.right = 2
        self.down = 3
        self.gameOver = False
        self.newTile()
        self.newTile()

    def newTile(self):
        emptySpots=[]
        for x in range(self.height):
            for y in range(self.width):
                if self.board[x][y] == 0:
                    emptySpots.append([x,y])
        if len(emptySpots) == 0:
            self.gameOver = True
            return
        randomSpot = emptySpots[np.random.choice(len(emptySpots), 1)[0]]
        self.board[randomSpot[0]][randomSpot[1]] = 1
        # there is a 10% chance that the new tile will be a 4 instead of a 2.
        if np.random.choice(10, 1) == 1:
            self.board[randomSpot[0]][randomSpot[1]] = 2

#note: rot90 is a built in numpy function which rotates a matrix 90 degrees counterclockwise.
    def performAction(self,action):
        tempBoard = self.board
        tempBoard = np.rot90(tempBoard, action)
        tempBoard = self.leftSlide(tempBoard)
        self.board = np.rot90(tempBoard, -action)
        self.newTile()

    def leftSlide(self, board):
        height = len(board)
        width = len(board[0])
        newboard = []
        for y in range(height):
            newrow = []
            prev = 0
            for x in board[y]:
                if x == prev and x != 0:
                    newrow = newrow[:-1]
                    newrow.append(x+1)
                    prev = 0
                elif x != 0:
                    newrow.append(x)
                    prev = x
                else:
                    continue
            newrow.extend([0 for i in range(width-len(newrow))])
            newboard.append(newrow)
        return np.array(newboard)

    def reset(self):
        self.board = np.zeros(shape=(self.height, self.width))
        self.left = 0
        self.up = 1
        self.right = 2
        self.down = 3
        self.gameOver = False
        self.newTile()
        self.newTile()

    def exponentiate(self):
        board = 2**self.board
        for i in range(self.height):
            for j in range(self.width):
                if board[i][j] == 1:
                    board[i][j] = 0
        return board

    def Max_number(self):
        return max(map(max, self.board))
