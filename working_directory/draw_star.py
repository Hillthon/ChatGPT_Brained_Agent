"""Draw a five-pointed star with Python's standard turtle module."""

import turtle


def draw_star(side_length: int = 180) -> None:
    """Draw a regular five-pointed star and keep the window open."""
    pen = turtle.Turtle()
    pen.color("gold")
    pen.pensize(4)
    pen.speed(3)

    for _ in range(5):
        pen.forward(side_length)
        pen.right(144)


def main() -> None:
    screen = turtle.Screen()
    screen.title("Five-Pointed Star")
    screen.bgcolor("midnight blue")
    draw_star()
    turtle.done()


if __name__ == "__main__":
    main()
