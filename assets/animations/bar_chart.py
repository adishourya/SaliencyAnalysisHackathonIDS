

from manim import *

font_style = dict(font_size=30, font="Simple Nerd Font",color=BLACK)

class BitBarChart(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        chart = BarChart(values=[78.9,93.8,97.3],
                         y_range=[0, 100, 20],
                         bar_names=["SmallConv","RESNET","VIT"],
                         # y_length=4,
                         x_length=4,
                     bar_colors=[BLUE,PINK,PURPLE])

        x = chart.get_x_axis()
        x.set_color(BLACK)

        y = chart.get_y_axis()
        y.set_color(BLACK)


        c_bar_lbls = chart.get_bar_labels(
            color=BLACK, label_constructor=MathTex, font_size=30
        )

        title= Text("Eval Accuracy",**font_style)
        title.next_to(chart,DOWN)
        self.add(title)

        self.play(Write(c_bar_lbls))
        self.play(Write(chart))

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 bar_chart.py BitBarChart")
