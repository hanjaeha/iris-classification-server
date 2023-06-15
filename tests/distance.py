import math
from typing import Any

class Point:
    """
    점을 2차원 좌표로 표현한다.
    >>> p_0 = Point()
    >>> p_1 = Point(3, 4)
    >>> p_0.calculate_distance(p_1)
    5.0
    """
    def __init__(self, x: float = 0, y: float = 0) -> None:
        """
        새 점의 위치를 초기화한다. x, y 좌표를 지정할 수 있다.
        그렇지 않으면 점은 기본적으로 원점에 설정된다.
        :param x: flaot x-coordinate
        :param y: flaot x-coordinate
        """
        self.move(x, y)

    def move(self, x: float, y: float) -> None:
        """
        점을 2차원 공간에서 새로운 위치로 이동한다.
        :param x: flaot x-coordinate
        :param y: flaot x-coordinate
        """
        self.x = x
        self.y = y

    def reset(self) -> None:
        """
        점을 기하학적 원점인 0, 0으로 재설정한다.
        """
        self.move(0, 0)

    def calculate_distance(self, other: Any) -> float:
        """
        현재 점에서 매개변수로 전달받은 두번째 점까지의
        유클리드 거리를 계산한다.
        :param other: Point instance
        :return: float distacne
        """
        return math.hypot(self.x - other.x, self.y - other.y)


p1 = Point()
p2 = Point()
p1.move(5, 3)
p2.move(10, 3)
print(p1.calculate_distance(p2))
