"""使用 turtle 绘制一个六边形。"""

import turtle


def draw_hexagon(size: int = 300) -> None:
    """绘制边长为 size 的正六边形。"""
    pen = turtle.Turtle()
    pen.speed("fastest")
    pen.pensize(3)
    pen.color("red")
    pen.fillcolor("red")

    pen.penup()
    pen.goto(-size / 2, size / 3)
    pen.setheading(-18)
    pen.pendown()

    pen.begin_fill()
    for _ in range(6):
        pen.forward(size)
        pen.right(60)
    pen.end_fill()

    pen.hideturtle()


if __name__ == "__main__":
    screen = turtle.Screen()
    screen.title("六边形")
    screen.bgcolor("white")
    draw_hexagon()
    screen.mainloop()
