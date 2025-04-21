import math
class Point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def __str__(self):
        return "({0},{1})".format(self.x, self.y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Point(x, y)
    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Point(x, y)
    def __mul__(self, other):
        x = self.x * other.x
        y = self.y * other.y
        return Point(x, y)
    def __len__(self):
        return math.sqrt(self.x**2+self.y**2)
    def __truediv__(self, other):
        if other.x == 0 or other.y == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")   
        x = self.x / other.x
        y = self.y / other.y
    def dot(self, other):
        return self.x * other.x + self.y * other.y