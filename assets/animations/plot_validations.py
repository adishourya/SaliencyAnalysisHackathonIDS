
from tkinter import BOTTOM
from manim import *

font_style0 = dict(font_size=10, font="Simple Nerd Font",color=GRAY)
font_style = dict(font_size=40, font="Simple Nerd Font",color=BLACK)

class RealLinePlot(Scene):
    def construct(self):
        # White background
        self.camera.background_color = WHITE


        # Points to plot
        points_vit = [
                (0,	0.9709702134132385),
                (1,	0.9734530448913574),
                (2,	0.9740260243415833),
                (3,	0.9726890921592712),
                (4,	0.9730710983276367),
                (5,	0.9732620716094971),
                (6,	0.9734530448913574),
                (7,	0.9738349914550781),
                (8,	0.9734530448913574),
                (9,	0.9738349914550781)
        ]

        points_resnet = [
            (0,	0.9409855008125305),
            (1,	0.9388846755027771),
            (2,	0.9428953528404236),
            (3,	0.9448052048683167),
            (4,	0.9432773590087891),
            (5,	0.9459511637687683),
            (6,	0.9482429623603821),
            (7,	0.941367506980896),
            (8,	0.9425134062767029),
            (9,	0.9388846755027771)
        ]

        points_simpleConv = [
            (0	,0.43124523758888245),
            (1,	0.6999618411064148),
            (2,	0.7446524500846863),
            (3,	0.730710506439209),
            (4,	0.7929717302322388),
            (5,	0.770053505897522)
        ]



        # Create axes with numbers
        axes = NumberPlane(
            x_range=[0,10,1],
            y_range=[0, 1, 0.20],
            x_length=10,
            y_length=10,
            axis_config={"color": BLUE, "include_numbers": True, "font_size":30},  # include numbers
            x_axis_config={"color": BLUE, "include_tip":False},
            y_axis_config={"color": BLUE, "include_tip": False},
        )
        axes_labels = axes.get_axis_labels(
            x_label=Text("Epochs", **font_style),
            y_label=Text("Valid Accuracy", **font_style)
        )

        x = axes.get_x_axis()
        x.numbers.set_color(BLACK)

        y = axes.get_y_axis()
        y.numbers.set_color(BLACK)

        self.add(axes, axes_labels)
        self.play(Write(axes),Write(axes_labels))


            
        # Add dots
        for x, y in points_simpleConv:
            dot = Dot(axes.coords_to_point(x, y),DEFAULT_DOT_RADIUS*2, color=GREEN)
            self.add(dot)

        # Connect points with lines
        for i in range(len(points_simpleConv)-1):
            line = Line(
                axes.coords_to_point(points_simpleConv[i][0], points_simpleConv[i][1]),
                axes.coords_to_point(points_simpleConv[i+1][0], points_simpleConv[i+1][1]),
                color=BLACK
            )
            self.play(Write(line))


        # Add dots
        for x, y in points_resnet:
            dot = Dot(axes.coords_to_point(x, y),DEFAULT_DOT_RADIUS*2, color=ORANGE)
            self.add(dot)

        # Connect points with lines
        for i in range(len(points_resnet)-1):
            line = Line(
                axes.coords_to_point(points_resnet[i][0], points_resnet[i][1]),
                axes.coords_to_point(points_resnet[i+1][0], points_resnet[i+1][1]),
                color=BLACK
            )
            self.play(Write(line))



        # Add dots
        for x, y in points_vit:
            dot = Dot(axes.coords_to_point(x, y),DEFAULT_DOT_RADIUS*2, color=PINK)
            self.add(dot)

        # Connect points with lines
        for i in range(len(points_vit)-1):
            line = Line(
                axes.coords_to_point(points_vit[i][0], points_vit[i][1]),
                axes.coords_to_point(points_vit[i+1][0], points_vit[i+1][1]),
                color=BLACK
            )
            self.play(Write(line))


if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1500,1500 plot_validations.py RealLinePlot")
