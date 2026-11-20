from pathlib import Path

import numpy as np
import pandas as pd

from manim import *


class TwoDimensionalGDConvergenceExample(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.a = 2
        self.b = -12
        self.c = 36

    def f(self, x: float) -> float:
        return 1 / 2 * self.a * x**2 + self.b * x + self.c

    def f_grad(self, x: float) -> float:
        return self.a * x + self.b

    def f_hess(self, _x: float) -> float:
        return self.a

    def get_gradient_descents_iterates(
        self, x_0: float, eta: float, num_iters: int
    ) -> list[tuple[float, float]]:
        iterates = [(x_0, self.f(x_0))]
        x = x_0

        for _ in range(num_iters):
            x = x - eta * self.f_grad(x)
            point = (x, self.f(x))
            iterates.append(point)

        return iterates

    def construct(self):
        self.next_section()
        consider = Tex(r"Consider for $a > 0$ and $b, c \in \mathbb{R}$", font_size=32)
        consider.to_corner(UL)
        self.play(Write(consider))

        f = MathTex(r"f(x) = \frac{1}{2} a x^2 + bx + c", font_size=32)
        self.play(Write(f))
        self.wait(1)

        self.play(f.animate.shift(UP / 2))
        f_grad = MathTex(r"f'(x) = a x + b", font_size=32)
        f_grad.shift(DOWN / 2)
        self.play(Write(f_grad))
        self.wait(1)

        self.play(f.animate.shift(UP / 2), f_grad.animate.shift(UP / 2))
        f_hess = MathTex(r"f''(x) = a", font_size=32)
        f_hess.shift(DOWN)
        self.play(Write(f_hess))
        self.wait(1)

        self.next_section()
        tmp_f = f.copy()
        tmp_f_grad = f_grad.copy()
        tmp_f_hess = f_hess.copy()

        tmp_f.next_to(consider, DOWN, coor_mask=(0, 1, 0))
        tmp_f_grad.next_to(tmp_f, DOWN)
        tmp_f_hess.next_to(tmp_f_grad, DOWN)

        self.play(
            f.animate.move_to(tmp_f),
            f_grad.animate.move_to(tmp_f_grad),
            f_hess.animate.move_to(tmp_f_hess),
        )

        recall_lecture_2 = Tex(r"Recall from lecture 2", font_size=32)
        recall_lecture_2.to_edge(LEFT)
        recall_lecture_2.next_to(f_hess, DOWN, coor_mask=(0, 1, 0))
        self.play(Write(recall_lecture_2))

        L_eqn1 = MathTex(r"L", r"\geq", r"\sigma_{\max}(\nabla^2 f)", font_size=32)
        L_eqn1.next_to(recall_lecture_2, DOWN, coor_mask=(0, 1, 0))
        self.play(Write(L_eqn1))
        self.wait(2)

        L_eqn2 = MathTex(r"L", r"\geq", r"a", font_size=32)
        L_eqn2.next_to(recall_lecture_2, DOWN, coor_mask=(0, 1, 0))
        self.play(TransformMatchingShapes(L_eqn1, L_eqn2))
        self.wait(2)

        L_eqn3 = MathTex(r"L", r"=", r"a", font_size=32).move_to(L_eqn1)
        L_eqn3.next_to(recall_lecture_2, DOWN, coor_mask=(0, 1, 0))
        self.play(TransformMatchingShapes(L_eqn2, L_eqn3))
        self.wait(2)

        self.next_section()
        recall_lecture_3 = Tex(
            r"Recall from lecture 3, the convergence condition for GD", font_size=32
        )
        recall_lecture_3.to_edge(LEFT)
        recall_lecture_3.next_to(L_eqn3, DOWN, coor_mask=(0, 1, 0))
        self.play(Write(recall_lecture_3))

        convergence_condition_eqn1 = MathTex(
            r"0", r"<", r"\eta", r"<", r"\frac{2}{L}", font_size=32
        )
        convergence_condition_eqn1.next_to(recall_lecture_3, DOWN, coor_mask=(0, 1, 0))
        self.play(Write(convergence_condition_eqn1))
        self.wait(1)

        convergence_condition_eqn2 = MathTex(
            r"L", r"<", r"\frac{2}{\eta}", font_size=32
        )
        convergence_condition_eqn2.next_to(recall_lecture_3, DOWN, coor_mask=(0, 1, 0))
        self.play(
            TransformMatchingShapes(
                convergence_condition_eqn1, convergence_condition_eqn2
            )
        )
        self.wait(1)

        convergence_condition_eqn3 = MathTex(
            r"a", r"<", r"\frac{2}{\eta}", font_size=32
        )
        convergence_condition_eqn3.next_to(recall_lecture_3, DOWN, coor_mask=(0, 1, 0))
        self.play(
            TransformMatchingShapes(
                convergence_condition_eqn2, convergence_condition_eqn3
            )
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(consider),
            Unwrite(f),
            Unwrite(f_grad),
            Unwrite(f_hess),
            Unwrite(recall_lecture_2),
            Unwrite(L_eqn3),
            Unwrite(recall_lecture_3),
            Unwrite(convergence_condition_eqn3),
        )

        ax = Axes(
            x_range=[0, 12, 1],
            y_range=[0, 6, 1],
            x_length=12,
            y_length=6,
        )
        self.play(Write(ax))

        graph = ax.plot(lambda x: self.f(x), x_range=[0, 10])
        label = ax.get_graph_label(
            graph,
            label=MathTex(r"f(x)", font_size=32),
            x_val=3,
            direction=UP,
        )
        self.play(Write(graph), Write(label))
        self.wait(1)

        gd_iterates = self.get_gradient_descents_iterates(x_0=4.5, eta=0.9, num_iters=5)
        gd_iterates = [ax.coords_to_point(x, y) for x, y in gd_iterates]

        starting_dot = Dot(gd_iterates[0], color=BLUE)
        self.play(GrowFromCenter(starting_dot))
        self.wait(1)

        convergence_vgroup = VGroup()
        for idx in range(1, len(gd_iterates)):
            prev_point = gd_iterates[idx - 1]
            next_point = gd_iterates[idx]

            traveling_dot = Dot(prev_point, color=GREEN)
            if idx == 1:
                self.play(GrowFromCenter(traveling_dot))
            else:
                self.add(traveling_dot)

            transition_arrow = Arrow(
                prev_point,
                next_point,
                color=GREEN,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.1,
            )
            self.play(
                traveling_dot.animate.move_to(next_point),
                GrowArrow(transition_arrow),
            )
            convergence_vgroup.add(traveling_dot, transition_arrow)

        self.wait(1)

        self.next_section()
        gd_iterates = self.get_gradient_descents_iterates(x_0=4.5, eta=1.1, num_iters=3)
        gd_iterates = [ax.coords_to_point(x, y) for x, y in gd_iterates]

        divergence_vgroup = VGroup()
        for idx in range(1, len(gd_iterates)):
            prev_point = gd_iterates[idx - 1]
            next_point = gd_iterates[idx]

            traveling_dot = Dot(prev_point, color=RED)
            if idx == 1:
                self.play(GrowFromCenter(traveling_dot))
            else:
                self.add(traveling_dot)

            transition_arrow = Arrow(
                prev_point,
                next_point,
                color=RED,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.1,
            )
            self.play(
                traveling_dot.animate.move_to(next_point),
                GrowArrow(transition_arrow),
            )
            divergence_vgroup.add(traveling_dot, transition_arrow)

        self.wait(1)

        self.next_section()

        self.play(
            Unwrite(ax),
            Unwrite(graph),
            Unwrite(label),
            Unwrite(starting_dot),
            Unwrite(convergence_vgroup),
            Unwrite(divergence_vgroup),
        )
        self.wait(1)


class ThreeDimensionalGDConvergenceExample(ThreeDScene):
    def f(self, x: float, y: float) -> float:
        return x**2 + 0.5 * y**2

    def f_polar(self, r: float, theta: float) -> float:
        return r**2 * (np.cos(theta) ** 2 + 0.5 * np.sin(theta) ** 2)

    def f_grad(self, x: float, y: float) -> np.ndarray:
        return np.array([2 * x, y])

    def f_hess(self, _x: float) -> float:
        return np.array([[2.0, 0.0], [0.0, 1.0]])

    def get_gradient_descents_iterates(
        self, x_0: float, y_0: float, eta: float, num_iters: int
    ) -> list[tuple[float, float, float]]:
        x, y = x_0, y_0

        iterates = [(x, y, self.f(x, y))]

        for _ in range(num_iters):
            grad = self.f_grad(x, y)
            x = x - eta * grad[0]
            y = y - eta * grad[1]

            iterates.append((x, y, self.f(x, y)))

        return iterates

    def construct(self):
        self.next_section()
        convergence_condition_eqn1 = MathTex(
            r"L", r"<", r"\frac{2}{\eta}", font_size=32
        )
        self.play(Write(convergence_condition_eqn1))
        self.wait(2)

        convergence_condition_eqn2 = MathTex(
            r"\sigma_{\max}", r"\nabla^2 f", r"<", r"\frac{2}{\eta}", font_size=32
        )
        self.play(
            TransformMatchingShapes(
                convergence_condition_eqn1, convergence_condition_eqn2
            )
        )
        self.wait(2)

        convergence_condition_eqn3 = MathTex(
            r"\lambda_{\max}", r"\nabla^2 f", r"<", r"\frac{2}{\eta}", font_size=32
        )
        self.play(
            TransformMatchingShapes(
                convergence_condition_eqn2, convergence_condition_eqn3
            )
        )
        self.wait(1)

        self.next_section()
        self.play(Unwrite(convergence_condition_eqn3))

        ax = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            z_range=[-5, 5, 2],
            x_length=10,
            y_length=10,
            z_length=10,
        )
        self.play(Create(ax))

        surface = Surface(
            lambda r, theta: ax.coords_to_point(
                r * np.cos(theta),
                r * np.sin(theta),
                self.f_polar(r, theta),
            ),
            u_range=[0, 2.5],
            v_range=[0, TAU],
            resolution=64,
            fill_opacity=0.7,
            checkerboard_colors=None,
        )

        self.move_camera(
            phi=35 * DEGREES,
            theta=-120 * DEGREES,
            focal_distance=10,
            frame_center=[0, 0, 2],
            added_anims=[Write(surface)],
        )
        self.wait(1)

        self.next_section()
        eigenvector = Arrow3D(
            start=ax.coords_to_point(0, 0, 0),
            end=ax.coords_to_point(2, 0, 0),
            color=YELLOW,
            resolution=48,
        )

        self.move_camera(
            phi=15 * DEGREES,
            theta=-135 * DEGREES,
            focal_distance=6,
            frame_center=[1, 0, 0],
            added_anims=[Write(eigenvector)],
        )
        self.wait(1)

        self.move_camera(
            phi=40 * DEGREES,
            theta=-100 * DEGREES,
            focal_distance=8,
            frame_center=[0, 0, 2],
        )

        self.next_section()
        gd_iterates = self.get_gradient_descents_iterates(
            x_0=1, y_0=1, eta=0.9, num_iters=5
        )
        gd_iterates = [ax.coords_to_point(x, y, z) for x, y, z in gd_iterates]

        starting_dot = Dot3D(gd_iterates[0], color=BLUE, resolution=16)
        self.play(GrowFromCenter(starting_dot))
        self.wait(1)

        convergence_vgroup = VGroup()
        for idx in range(1, len(gd_iterates)):
            prev_point = gd_iterates[idx - 1]
            next_point = gd_iterates[idx]

            traveling_dot = Dot3D(prev_point, color=GREEN, resolution=16)
            if idx == 1:
                self.play(GrowFromCenter(traveling_dot))
            else:
                self.add(traveling_dot)

            transition_arrow = Arrow(
                prev_point,
                next_point,
                color=GREEN,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.1,
            )
            self.play(
                traveling_dot.animate.move_to(next_point),
                GrowArrow(transition_arrow),
            )
            convergence_vgroup.add(traveling_dot, transition_arrow)

        self.wait(1)

        self.next_section()
        gd_iterates = self.get_gradient_descents_iterates(
            x_0=1, y_0=1, eta=1.1, num_iters=5
        )
        gd_iterates = [ax.coords_to_point(x, y, z) for x, y, z in gd_iterates]

        divergence_vgroup = VGroup()
        for idx in range(1, len(gd_iterates)):
            prev_point = gd_iterates[idx - 1]
            next_point = gd_iterates[idx]

            traveling_dot = Dot3D(prev_point, color=RED, resolution=16)
            if idx == 1:
                self.play(GrowFromCenter(traveling_dot))
            else:
                self.add(traveling_dot)

            transition_arrow = Arrow(
                prev_point,
                next_point,
                color=RED,
                stroke_width=4,
                max_tip_length_to_length_ratio=0.1,
            )
            self.play(
                traveling_dot.animate.move_to(next_point),
                GrowArrow(transition_arrow),
            )
            divergence_vgroup.add(traveling_dot, transition_arrow)

        self.wait(1)

        self.next_section()
        self.move_camera(
            phi=0,
            theta=0,
            focal_distance=10,
            frame_center=[0, 0, 0],
            added_anims=[
                Unwrite(ax),
                Unwrite(surface),
                Unwrite(eigenvector),
                Unwrite(starting_dot),
                Unwrite(convergence_vgroup),
                Unwrite(divergence_vgroup),
            ],
        )
        self.wait(1)


class SharpnessScene(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.sharpness_logs = {}
        for log_dir in Path("../logs").iterdir():
            metrics_path = log_dir / "version_0" / "metrics.csv"
            self.sharpness_logs[log_dir.name] = self.read_sharpness(metrics_path)

        self.precond_sharpness_logs = {}
        for log_dir in Path("../logs").iterdir():
            if "adam" not in log_dir.name:
                continue

            metrics_path = log_dir / "version_0" / "metrics.csv"
            self.precond_sharpness_logs[log_dir.name] = self.read_precond_sharpness(
                metrics_path
            )

        self.raw_sharpness_logs = {}
        for log_dir in Path("../logs").iterdir():
            if "adam" not in log_dir.name:
                continue

            metrics_path = log_dir / "version_0" / "metrics.csv"
            self.raw_sharpness_logs[log_dir.name] = self.read_raw_sharpness(
                metrics_path
            )

        self.loss_logs = {}
        for log_dir in Path("../logs").iterdir():
            metrics_path = log_dir / "version_0" / "metrics.csv"
            self.loss_logs[log_dir.name] = self.read_loss(metrics_path)

    def read_sharpness(self, path: str) -> list[float]:
        df = pd.read_csv(path)
        mask = df["Sharpness"].notna()
        sharpness = df.loc[mask, "Sharpness"].tolist()
        return sharpness

    def get_sharpness_axes(self, log_keys: list[str]) -> Axes:
        max_x = max(len(self.sharpness_logs[log_key]) for log_key in log_keys)
        max_y = max(max(self.sharpness_logs[log_key]) for log_key in log_keys)

        ax = Axes(
            x_range=[0, max_x, 250],
            y_range=[0, max_y, 10],
            tips=False,
            axis_config={"include_numbers": True, "font_size": 16},
        )

        return ax

    def get_sharpness_dots(
        self,
        ax: Axes,
        sharpness: list[float],
        color: ParsableManimColor,
        fill_opacity: float = 1.0,
    ) -> VGroup:
        dots = VGroup()

        for step, sharpness_value in enumerate(sharpness):
            point = ax.coords_to_point(step, sharpness_value)
            dot = Dot(point, color=color, radius=0.02, fill_opacity=fill_opacity)
            dots.add(dot)

        return dots

    def get_sharpness_threshold(
        self,
        ax: Axes,
        eta: float,
        mathtex: str,
        color: ParsableManimColor,
    ) -> VGroup:
        threshold_line = DashedLine(
            ax.coords_to_point(0, 2 / eta),
            ax.coords_to_point(ax.x_range[1], 2 / eta),
            stroke_width=4,
            color=color,
        )
        threshold_label = MathTex(mathtex, font_size=16, color=color)
        threshold_label.move_to(ax.coords_to_point(ax.x_range[1], 2 / eta))
        threshold_label.shift(RIGHT / 2)

        return VGroup(threshold_line, threshold_label)

    def read_precond_sharpness(self, path: str) -> list[float]:
        df = pd.read_csv(path)
        mask = df["Preconditioned Sharpness"].notna()
        sharpness = df.loc[mask, "Preconditioned Sharpness"].tolist()

        sharpness = [abs(s) for s in sharpness]
        sharpness = [s for s in sharpness if s != 0]

        return sharpness

    def read_raw_sharpness(self, path: str) -> list[float]:
        df = pd.read_csv(path)
        if "Sharpness" not in df.columns:
            return []
        mask = df["Sharpness"].notna()
        sharpness = df.loc[mask, "Sharpness"].tolist()

        sharpness = [abs(s) for s in sharpness]
        sharpness = [s for s in sharpness if s != 0]

        return sharpness

    def get_precond_sharpness_axes(self, log_keys: list[str]) -> Axes:
        max_x = max(len(self.precond_sharpness_logs[k]) for k in log_keys)

        max_y = max(max(self.precond_sharpness_logs[k]) for k in log_keys)
        min_y = min(min(self.precond_sharpness_logs[k]) for k in log_keys)

        exp_min = np.floor(np.log10(min_y))
        exp_max = np.ceil(np.log10(max_y))

        ax = Axes(
            x_range=[0, max_x, 250],
            y_range=[exp_min, exp_max, 1],
            tips=False,
            axis_config={"include_numbers": True, "font_size": 16},
            y_axis_config={"scaling": LogBase(base=10, custom_labels=True)},
        )

        return ax

    def get_raw_sharpness_axes(self, log_keys: list[str]) -> Axes:
        max_x = max(len(self.raw_sharpness_logs[k]) for k in log_keys)

        max_y = max(max(self.raw_sharpness_logs[k]) for k in log_keys)
        min_y = min(min(self.raw_sharpness_logs[k]) for k in log_keys)

        exp_min = np.floor(np.log10(min_y))
        exp_max = np.ceil(np.log10(max_y))

        ax = Axes(
            x_range=[0, max_x, 250],
            y_range=[exp_min, exp_max, 1],
            tips=False,
            axis_config={"include_numbers": True, "font_size": 16},
            y_axis_config={"scaling": LogBase(base=10, custom_labels=True)},
        )

        return ax

    def get_precond_sharpness_dots(
        self,
        ax: Axes,
        precond_sharpness: list[float],
        color: ParsableManimColor,
        fill_opacity: float = 1.0,
    ) -> VGroup:
        dots = VGroup()

        for step, precond_sharpness_value in enumerate(precond_sharpness):
            point = ax.coords_to_point(step, precond_sharpness_value)
            dot = Dot(point, color=color, radius=0.02, fill_opacity=fill_opacity)
            dots.add(dot)

        return dots

    def get_precond_sharpness_threshold(
        self,
        ax: Axes,
        eta: float,
        mathtex: str,
        color: ParsableManimColor,
    ) -> VGroup:
        threshold_line = DashedLine(
            ax.coords_to_point(0, 38 / eta),
            ax.coords_to_point(ax.x_range[1], 38 / eta),
            stroke_width=4,
            color=color,
        )
        threshold_label = MathTex(mathtex, font_size=16, color=color)
        threshold_label.move_to(ax.coords_to_point(ax.x_range[1], 38 / eta))
        threshold_label.shift(RIGHT / 2)

        return VGroup(threshold_line, threshold_label)

    def read_loss(self, path: str) -> list[float]:
        df = pd.read_csv(path)
        mask = df["Training Loss"].notna()
        loss = df.loc[mask, "Training Loss"].tolist()
        return loss

    def get_loss_axes(self, log_keys: list[str]) -> Axes:
        max_x = max(len(self.loss_logs[log_key]) for log_key in log_keys)
        max_y = max(max(self.loss_logs[log_key]) for log_key in log_keys)

        ax = Axes(x_range=[0, max_x, 1], y_range=[0, max_y, 1])
        return ax

    def get_loss_plot(
        self,
        log_keys: list[str],
    ) -> VGroup:
        colours = [BLUE, GREEN, GOLD, MAROON]
        loss_plots = VGroup()

        ax = self.get_loss_axes(log_keys)

        for i, log_key in enumerate(log_keys):
            loss = self.loss_logs[log_key]
            plot = ax.plot_line_graph(
                x_values=list(range(len(loss))),
                y_values=loss,
                line_color=colours[i],
                add_vertex_dots=False,
                stroke_opacity=0.2,
            )
            loss_plots.add(plot)

        return loss_plots

    def get_eps_label(
        self, ax: Axes, sharpness: list[float], mathtex: str, color: ParsableManimColor
    ) -> MathTex:
        eps_label = MathTex(mathtex, font_size=16, color=color)
        eps_label.move_to(ax.coords_to_point(len(sharpness), sharpness[-1]))
        eps_label.shift(RIGHT / 2)

        return eps_label


class SGDSharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_sharpness_axes(
            [
                "lr_0.1_opt_sgd",
                "lr_0.04_opt_sgd",
                "lr_0.025_opt_sgd",
                "lr_0.01818181818181818_opt_sgd",
            ]
        )
        self.play(Write(ax))

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(r"SGD Sharpness", font_size=24)
        title.to_edge(UP)

        sharpness_1 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.1_opt_sgd"],
            color=BLUE,
        )
        threshold_1 = self.get_sharpness_threshold(
            ax,
            eta=0.1,
            mathtex=r"\eta = \frac{2}{20}",
            color=BLUE,
        )

        sharpness_04 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.04_opt_sgd"],
            color=GREEN,
        )
        threshold_04 = self.get_sharpness_threshold(
            ax,
            eta=0.04,
            mathtex=r"\eta = \frac{2}{50}",
            color=GREEN,
        )

        sharpness_025 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.025_opt_sgd"],
            color=GOLD,
        )
        threshold_025 = self.get_sharpness_threshold(
            ax,
            eta=0.025,
            mathtex=r"\eta = \frac{2}{80}",
            color=GOLD,
        )

        sharpness_018 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.01818181818181818_opt_sgd"],
            color=MAROON,
        )
        threshold_018 = self.get_sharpness_threshold(
            ax,
            eta=0.01818181818181818,
            mathtex=r"\eta = \frac{2}{110}",
            color=MAROON,
        )

        loss = self.get_loss_plot(
            [
                "lr_0.1_opt_sgd",
                "lr_0.04_opt_sgd",
                "lr_0.025_opt_sgd",
                "lr_0.01818181818181818_opt_sgd",
            ]
        )
        loss.set_z_index(0)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(sharpness_1),
            Write(threshold_1),
            Write(sharpness_04),
            Write(threshold_04),
            Write(sharpness_025),
            Write(threshold_025),
            Write(sharpness_018),
            Write(threshold_018),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(sharpness_1),
            Unwrite(threshold_1),
            Unwrite(sharpness_04),
            Unwrite(threshold_04),
            Unwrite(sharpness_025),
            Unwrite(threshold_025),
            Unwrite(sharpness_018),
            Unwrite(threshold_018),
            Unwrite(loss),
        )
        self.wait(1)


class DPSGD1Sharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_sharpness_axes(
            [
                "lr_0.1_opt_sgd",
                "lr_0.1_opt_sgd_eps_16_mgn_3.0",
                "lr_0.1_opt_sgd_eps_32_mgn_3.0",
                "lr_0.1_opt_sgd_eps_64_mgn_3.0",
            ]
        )
        self.play(Write(ax))

        sharpness_1 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.1_opt_sgd"],
            color=BLUE,
            fill_opacity=0.5,
        )
        threshold_1 = self.get_sharpness_threshold(
            ax,
            eta=0.1,
            mathtex=r"\eta = \frac{2}{20}",
            color=BLUE,
        )

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(
            r"DP-SGD Sharpness vs. $\varepsilon$ ($\eta=\frac{2}{20}$)",
            font_size=24,
        )
        title.to_edge(UP)

        sharpness_1_16 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.1_opt_sgd_eps_16_mgn_3.0"],
            color=GREEN,
        )
        eps_label_1_16 = self.get_eps_label(
            ax,
            self.sharpness_logs["lr_0.1_opt_sgd_eps_16_mgn_3.0"],
            mathtex=r"\varepsilon = 16",
            color=GREEN,
        )
        eps_label_1_16.shift(UP / 4)

        sharpness_1_32 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.1_opt_sgd_eps_32_mgn_3.0"],
            color=GOLD,
        )
        eps_label_1_32 = self.get_eps_label(
            ax,
            self.sharpness_logs["lr_0.1_opt_sgd_eps_32_mgn_3.0"],
            mathtex=r"\varepsilon = 32",
            color=GOLD,
        )

        sharpness_1_64 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.1_opt_sgd_eps_64_mgn_3.0"],
            color=MAROON,
        )
        eps_label_1_64 = self.get_eps_label(
            ax,
            self.sharpness_logs["lr_0.1_opt_sgd_eps_64_mgn_3.0"],
            mathtex=r"\varepsilon = 64",
            color=MAROON,
        )

        loss = self.get_loss_plot(
            [
                "lr_0.1_opt_sgd",
                "lr_0.1_opt_sgd_eps_16_mgn_3.0",
                "lr_0.1_opt_sgd_eps_32_mgn_3.0",
                "lr_0.1_opt_sgd_eps_64_mgn_3.0",
            ]
        )
        loss.set_z_index(0)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(sharpness_1),
            Write(threshold_1),
            Write(sharpness_1_16),
            Write(eps_label_1_16),
            Write(sharpness_1_32),
            Write(eps_label_1_32),
            Write(sharpness_1_64),
            Write(eps_label_1_64),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(sharpness_1),
            Unwrite(threshold_1),
            Unwrite(sharpness_1_16),
            Unwrite(eps_label_1_16),
            Unwrite(sharpness_1_32),
            Unwrite(eps_label_1_32),
            Unwrite(sharpness_1_64),
            Unwrite(eps_label_1_64),
            Unwrite(loss),
        )
        self.wait(1)


class DPSGD04Sharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_sharpness_axes(
            [
                "lr_0.04_opt_sgd",
                "lr_0.04_opt_sgd_eps_16_mgn_3.0",
                "lr_0.04_opt_sgd_eps_32_mgn_3.0",
                "lr_0.04_opt_sgd_eps_64_mgn_3.0",
            ]
        )
        self.play(Write(ax))

        sharpness_04 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.04_opt_sgd"],
            color=BLUE,
            fill_opacity=0.5,
        )
        threshold_04 = self.get_sharpness_threshold(
            ax,
            eta=0.04,
            mathtex=r"\eta = \frac{2}{50}",
            color=BLUE,
        )

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(
            r"DP-SGD Sharpness vs. $\varepsilon$ ($\eta=\frac{2}{50}$)",
            font_size=24,
        )
        title.to_edge(UP)

        sharpness_04_16 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.04_opt_sgd_eps_16_mgn_3.0"],
            color=GREEN,
        )
        eps_label_04_16 = self.get_eps_label(
            ax,
            self.sharpness_logs["lr_0.04_opt_sgd_eps_16_mgn_3.0"],
            mathtex=r"\varepsilon = 16",
            color=GREEN,
        )

        sharpness_04_32 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.04_opt_sgd_eps_32_mgn_3.0"],
            color=GOLD,
        )
        eps_label_04_32 = self.get_eps_label(
            ax,
            self.sharpness_logs["lr_0.04_opt_sgd_eps_32_mgn_3.0"],
            mathtex=r"\varepsilon = 32",
            color=GOLD,
        )

        sharpness_04_64 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.04_opt_sgd_eps_64_mgn_3.0"],
            color=MAROON,
        )
        eps_label_04_64 = self.get_eps_label(
            ax,
            self.sharpness_logs["lr_0.04_opt_sgd_eps_64_mgn_3.0"],
            mathtex=r"\varepsilon = 64",
            color=MAROON,
        )

        loss = self.get_loss_plot(
            [
                "lr_0.04_opt_sgd",
                "lr_0.04_opt_sgd_eps_16_mgn_3.0",
                "lr_0.04_opt_sgd_eps_32_mgn_3.0",
                "lr_0.04_opt_sgd_eps_64_mgn_3.0",
            ]
        )
        loss.set_z_index(0)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(sharpness_04),
            Write(threshold_04),
            Write(sharpness_04_16),
            Write(eps_label_04_16),
            Write(sharpness_04_32),
            Write(eps_label_04_32),
            Write(sharpness_04_64),
            Write(eps_label_04_64),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(sharpness_04),
            Unwrite(threshold_04),
            Unwrite(sharpness_04_16),
            Unwrite(eps_label_04_16),
            Unwrite(sharpness_04_32),
            Unwrite(eps_label_04_32),
            Unwrite(sharpness_04_64),
            Unwrite(eps_label_04_64),
            Unwrite(loss),
        )
        self.wait(1)


class DPSGD025Sharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_sharpness_axes(
            [
                "lr_0.025_opt_sgd",
                "lr_0.025_opt_sgd_eps_16_mgn_3.0",
                "lr_0.025_opt_sgd_eps_32_mgn_3.0",
                "lr_0.025_opt_sgd_eps_64_mgn_3.0",
            ]
        )
        self.play(Write(ax))

        sharpness_025 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.025_opt_sgd"],
            color=BLUE,
            fill_opacity=0.5,
        )
        threshold_025 = self.get_sharpness_threshold(
            ax,
            eta=0.025,
            mathtex=r"\eta = \frac{2}{80}",
            color=BLUE,
        )

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(
            r"DP-SGD Sharpness vs. $\varepsilon$ ($\eta=\frac{2}{80}$)",
            font_size=24,
        )
        title.to_edge(UP)

        sharpness_025_16 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.025_opt_sgd_eps_16_mgn_3.0"],
            color=GREEN,
        )
        eps_label_025_16 = self.get_eps_label(
            ax,
            self.sharpness_logs["lr_0.025_opt_sgd_eps_16_mgn_3.0"],
            mathtex=r"\varepsilon = 16",
            color=GREEN,
        )

        sharpness_025_32 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.025_opt_sgd_eps_32_mgn_3.0"],
            color=GOLD,
        )
        eps_label_025_32 = self.get_eps_label(
            ax,
            self.sharpness_logs["lr_0.025_opt_sgd_eps_32_mgn_3.0"],
            mathtex=r"\varepsilon = 32",
            color=GOLD,
        )

        sharpness_025_64 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.025_opt_sgd_eps_64_mgn_3.0"],
            color=MAROON,
        )
        eps_label_025_64 = self.get_eps_label(
            ax,
            self.sharpness_logs["lr_0.025_opt_sgd_eps_64_mgn_3.0"],
            mathtex=r"\varepsilon = 64",
            color=MAROON,
        )

        loss = self.get_loss_plot(
            [
                "lr_0.025_opt_sgd",
                "lr_0.025_opt_sgd_eps_16_mgn_3.0",
                "lr_0.025_opt_sgd_eps_32_mgn_3.0",
                "lr_0.025_opt_sgd_eps_64_mgn_3.0",
            ]
        )
        loss.set_z_index(0)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(sharpness_025),
            Write(threshold_025),
            Write(sharpness_025_16),
            Write(eps_label_025_16),
            Write(sharpness_025_32),
            Write(eps_label_025_32),
            Write(sharpness_025_64),
            Write(eps_label_025_64),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(sharpness_025),
            Unwrite(threshold_025),
            Unwrite(sharpness_025_16),
            Unwrite(eps_label_025_16),
            Unwrite(sharpness_025_32),
            Unwrite(eps_label_025_32),
            Unwrite(sharpness_025_64),
            Unwrite(eps_label_025_64),
            Unwrite(loss),
        )
        self.wait(1)


class DPSGD018Sharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_sharpness_axes(
            [
                "lr_0.01818181818181818_opt_sgd",
                "lr_0.01818181818181818_opt_sgd_eps_16_mgn_3.0",
                "lr_0.01818181818181818_opt_sgd_eps_32_mgn_3.0",
                "lr_0.01818181818181818_opt_sgd_eps_64_mgn_3.0",
            ]
        )
        self.play(Write(ax))

        sharpness_018 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.01818181818181818_opt_sgd"],
            color=BLUE,
            fill_opacity=0.5,
        )
        threshold_018 = self.get_sharpness_threshold(
            ax,
            eta=0.01818181818181818,
            mathtex=r"\eta = \frac{2}{110}",
            color=BLUE,
        )

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(
            r"DP-SGD Sharpness vs. $\varepsilon$ ($\eta=\frac{2}{110}$)",
            font_size=24,
        )
        title.to_edge(UP)

        sharpness_018_16 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.01818181818181818_opt_sgd_eps_16_mgn_3.0"],
            color=GREEN,
        )
        eps_label_018_16 = self.get_eps_label(
            ax,
            self.sharpness_logs["lr_0.01818181818181818_opt_sgd_eps_16_mgn_3.0"],
            mathtex=r"\varepsilon = 16",
            color=GREEN,
        )

        sharpness_018_32 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.01818181818181818_opt_sgd_eps_32_mgn_3.0"],
            color=GOLD,
        )
        eps_label_018_32 = self.get_eps_label(
            ax,
            self.sharpness_logs["lr_0.01818181818181818_opt_sgd_eps_32_mgn_3.0"],
            mathtex=r"\varepsilon = 32",
            color=GOLD,
        )

        sharpness_018_64 = self.get_sharpness_dots(
            ax,
            self.sharpness_logs["lr_0.01818181818181818_opt_sgd_eps_64_mgn_3.0"],
            color=MAROON,
        )
        eps_label_018_64 = self.get_eps_label(
            ax,
            self.sharpness_logs["lr_0.01818181818181818_opt_sgd_eps_64_mgn_3.0"],
            mathtex=r"\varepsilon = 64",
            color=MAROON,
        )
        eps_label_018_64.shift(UP / 4)

        loss = self.get_loss_plot(
            [
                "lr_0.01818181818181818_opt_sgd",
                "lr_0.01818181818181818_opt_sgd_eps_16_mgn_3.0",
                "lr_0.01818181818181818_opt_sgd_eps_32_mgn_3.0",
                "lr_0.01818181818181818_opt_sgd_eps_64_mgn_3.0",
            ]
        )
        loss.set_z_index(0)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(sharpness_018),
            Write(threshold_018),
            Write(sharpness_018_16),
            Write(eps_label_018_16),
            Write(sharpness_018_32),
            Write(eps_label_018_32),
            Write(sharpness_018_64),
            Write(eps_label_018_64),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(sharpness_018),
            Unwrite(threshold_018),
            Unwrite(sharpness_018_16),
            Unwrite(eps_label_018_16),
            Unwrite(sharpness_018_32),
            Unwrite(eps_label_018_32),
            Unwrite(sharpness_018_64),
            Unwrite(eps_label_018_64),
            Unwrite(loss),
        )
        self.wait(1)


class AdamSharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_precond_sharpness_axes(
            [
                "lr_0.001_opt_adam",
                "lr_0.00032_opt_adam",
                "lr_0.0001_opt_adam",
                "lr_3e-05_opt_adam",
            ]
        )
        self.play(Write(ax))

        precond_sharpness_001 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.001_opt_adam"],
            color=BLUE,
        )
        precond_threshold_001 = self.get_precond_sharpness_threshold(
            ax,
            eta=0.001,
            mathtex=r"\eta = 1\text{e-}3",
            color=BLUE,
        )

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Preconditioned Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(r"Adam Preconditioned Sharpness", font_size=24)
        title.to_edge(UP)

        precond_sharpness_000032 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.00032_opt_adam"],
            color=GREEN,
        )
        precond_threshold_000032 = self.get_precond_sharpness_threshold(
            ax,
            eta=0.00032,
            mathtex=r"\eta = 3.2\text{e-}4",
            color=GREEN,
        )

        precond_sharpness_0001 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.0001_opt_adam"],
            color=GOLD,
        )
        precond_threshold_0001 = self.get_precond_sharpness_threshold(
            ax,
            eta=0.0001,
            mathtex=r"\eta = 1\text{e-}4",
            color=GOLD,
        )

        precond_sharpness_00003 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_3e-05_opt_adam"],
            color=MAROON,
        )
        precond_threshold_00003 = self.get_precond_sharpness_threshold(
            ax,
            eta=3e-05,
            mathtex=r"\eta = 3\text{e-}5",
            color=MAROON,
        )

        loss = self.get_loss_plot(
            [
                "lr_0.001_opt_adam",
                "lr_0.00032_opt_adam",
                "lr_0.0001_opt_adam",
                "lr_3e-05_opt_adam",
            ]
        )
        loss.set_z_index(0)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(precond_sharpness_001),
            Write(precond_threshold_001),
            Write(precond_sharpness_000032),
            Write(precond_threshold_000032),
            Write(precond_sharpness_0001),
            Write(precond_threshold_0001),
            Write(precond_sharpness_00003),
            Write(precond_threshold_00003),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(precond_sharpness_001),
            Unwrite(precond_threshold_001),
            Unwrite(precond_sharpness_000032),
            Unwrite(precond_threshold_000032),
            Unwrite(precond_sharpness_0001),
            Unwrite(precond_threshold_0001),
            Unwrite(precond_sharpness_00003),
            Unwrite(precond_threshold_00003),
            Unwrite(loss),
        )
        self.wait(1)


class AdamRawSharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_raw_sharpness_axes(
            [
                "lr_0.001_opt_adam",
                "lr_0.00032_opt_adam",
                "lr_0.0001_opt_adam",
                "lr_3e-05_opt_adam",
            ]
        )
        self.play(Write(ax))

        raw_sharpness_001 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.001_opt_adam"],
            color=BLUE,
        )
        raw_sharpness_00032 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.00032_opt_adam"],
            color=GREEN,
        )
        raw_sharpness_0001 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.0001_opt_adam"],
            color=GOLD,
        )
        raw_sharpness_00003 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_3e-05_opt_adam"],
            color=MAROON,
        )
        loss = self.get_loss_plot(
            [
                "lr_0.001_opt_adam",
                "lr_0.00032_opt_adam",
                "lr_0.0001_opt_adam",
                "lr_3e-05_opt_adam",
            ]
        )

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Raw Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(r"Adam Raw Sharpness", font_size=24)
        title.to_edge(UP)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(raw_sharpness_001),
            Write(raw_sharpness_00032),
            Write(raw_sharpness_0001),
            Write(raw_sharpness_00003),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(raw_sharpness_001),
            Unwrite(raw_sharpness_00032),
            Unwrite(raw_sharpness_0001),
            Unwrite(raw_sharpness_00003),
            Unwrite(loss),
        )
        self.wait(1)


class DPAdam001Sharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_precond_sharpness_axes(
            [
                "lr_0.001_opt_adam",
                "lr_0.001_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_0.001_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_0.001_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        self.play(Write(ax))

        precond_sharpness_001 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.001_opt_adam"],
            color=BLUE,
            fill_opacity=0.5,
        )
        precond_threshold_001 = self.get_precond_sharpness_threshold(
            ax,
            eta=0.001,
            mathtex=r"\eta = 1\text{e-}3",
            color=BLUE,
        )

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Preconditioned Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(r"DP-Adam Precond Sharpness ($\eta=1\text{e-}3$)", font_size=24)
        title.to_edge(UP)

        precond_sharpness_001_16 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.001_opt_adam_eps_16_mgn_3.0_bc_False"],
            color=GREEN,
        )
        eps_label_001_16 = self.get_eps_label(
            ax,
            self.precond_sharpness_logs["lr_0.001_opt_adam_eps_16_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 16",
            color=GREEN,
        )

        precond_sharpness_001_32 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.001_opt_adam_eps_32_mgn_3.0_bc_False"],
            color=GOLD,
        )
        eps_label_001_32 = self.get_eps_label(
            ax,
            self.precond_sharpness_logs["lr_0.001_opt_adam_eps_32_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 32",
            color=GOLD,
        )

        precond_sharpness_001_64 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.001_opt_adam_eps_64_mgn_3.0_bc_False"],
            color=MAROON,
        )
        eps_label_001_64 = self.get_eps_label(
            ax,
            self.precond_sharpness_logs["lr_0.001_opt_adam_eps_64_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 64",
            color=MAROON,
        )

        loss = self.get_loss_plot(
            [
                "lr_0.001_opt_adam",
                "lr_0.001_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_0.001_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_0.001_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        loss.set_z_index(0)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(precond_sharpness_001),
            Write(precond_threshold_001),
            Write(precond_sharpness_001_16),
            Write(eps_label_001_16),
            Write(precond_sharpness_001_32),
            Write(eps_label_001_32),
            Write(precond_sharpness_001_64),
            Write(eps_label_001_64),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(precond_sharpness_001),
            Unwrite(precond_threshold_001),
            Unwrite(precond_sharpness_001_16),
            Unwrite(eps_label_001_16),
            Unwrite(precond_sharpness_001_32),
            Unwrite(eps_label_001_32),
            Unwrite(precond_sharpness_001_64),
            Unwrite(eps_label_001_64),
            Unwrite(loss),
        )
        self.wait(1)


class DPAdam001RawSharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_raw_sharpness_axes(
            [
                "lr_0.001_opt_adam",
                "lr_0.001_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_0.001_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_0.001_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        self.play(Write(ax))

        raw_sharpness_001 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.001_opt_adam"],
            color=BLUE,
            fill_opacity=0.5,
        )

        raw_sharpness_001_16 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.001_opt_adam_eps_16_mgn_3.0_bc_False"],
            color=GREEN,
        )
        raw_eps_label_001_16 = self.get_eps_label(
            ax,
            self.raw_sharpness_logs["lr_0.001_opt_adam_eps_16_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 16",
            color=GREEN,
        )

        raw_sharpness_001_32 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.001_opt_adam_eps_32_mgn_3.0_bc_False"],
            color=GOLD,
        )
        raw_eps_label_001_32 = self.get_eps_label(
            ax,
            self.raw_sharpness_logs["lr_0.001_opt_adam_eps_32_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 32",
            color=GOLD,
        )

        raw_sharpness_001_64 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.001_opt_adam_eps_64_mgn_3.0_bc_False"],
            color=MAROON,
        )
        raw_eps_label_001_64 = self.get_eps_label(
            ax,
            self.raw_sharpness_logs["lr_0.001_opt_adam_eps_64_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 64",
            color=MAROON,
        )
        raw_eps_label_001_64.shift(UP / 4)

        loss = self.get_loss_plot(
            [
                "lr_0.001_opt_adam",
                "lr_0.001_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_0.001_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_0.001_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        loss.set_z_index(0)

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Raw Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(r"DP-Adam Raw Sharpness ($\eta=1\text{e-}3$)", font_size=24)
        title.to_edge(UP)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(raw_sharpness_001),
            Write(raw_sharpness_001_16),
            Write(raw_eps_label_001_16),
            Write(raw_sharpness_001_32),
            Write(raw_eps_label_001_32),
            Write(raw_sharpness_001_64),
            Write(raw_eps_label_001_64),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(raw_sharpness_001),
            Unwrite(raw_sharpness_001_16),
            Unwrite(raw_eps_label_001_16),
            Unwrite(raw_sharpness_001_32),
            Unwrite(raw_eps_label_001_32),
            Unwrite(raw_sharpness_001_64),
            Unwrite(raw_eps_label_001_64),
            Unwrite(loss),
        )
        self.wait(1)


class DPAdam00032Sharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_precond_sharpness_axes(
            [
                "lr_0.00032_opt_adam",
                "lr_0.00032_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_0.00032_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_0.00032_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        self.play(Write(ax))

        precond_sharpness_00032 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.00032_opt_adam"],
            color=BLUE,
            fill_opacity=0.5,
        )
        precond_threshold_00032 = self.get_precond_sharpness_threshold(
            ax,
            eta=0.00032,
            mathtex=r"\eta = 3.2\text{e-}4",
            color=BLUE,
        )

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Preconditioned Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(r"DP-Adam Precond Sharpness ($\eta=3.2\text{e-}4$)", font_size=24)
        title.to_edge(UP)

        precond_sharpness_00032_16 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.00032_opt_adam_eps_16_mgn_3.0_bc_False"],
            color=GREEN,
        )
        eps_label_00032_16 = self.get_eps_label(
            ax,
            self.precond_sharpness_logs["lr_0.00032_opt_adam_eps_16_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 16",
            color=GREEN,
        )

        precond_sharpness_00032_32 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.00032_opt_adam_eps_32_mgn_3.0_bc_False"],
            color=GOLD,
        )
        eps_label_00032_32 = self.get_eps_label(
            ax,
            self.precond_sharpness_logs["lr_0.00032_opt_adam_eps_32_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 32",
            color=GOLD,
        )

        precond_sharpness_00032_64 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.00032_opt_adam_eps_64_mgn_3.0_bc_False"],
            color=MAROON,
        )
        eps_label_00032_64 = self.get_eps_label(
            ax,
            self.precond_sharpness_logs["lr_0.00032_opt_adam_eps_64_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 64",
            color=MAROON,
        )

        loss = self.get_loss_plot(
            [
                "lr_0.00032_opt_adam",
                "lr_0.00032_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_0.00032_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_0.00032_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        loss.set_z_index(0)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(precond_sharpness_00032),
            Write(precond_threshold_00032),
            Write(precond_sharpness_00032_16),
            Write(eps_label_00032_16),
            Write(precond_sharpness_00032_32),
            Write(eps_label_00032_32),
            Write(precond_sharpness_00032_64),
            Write(eps_label_00032_64),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(precond_sharpness_00032),
            Unwrite(precond_threshold_00032),
            Unwrite(precond_sharpness_00032_16),
            Unwrite(eps_label_00032_16),
            Unwrite(precond_sharpness_00032_32),
            Unwrite(eps_label_00032_32),
            Unwrite(precond_sharpness_00032_64),
            Unwrite(eps_label_00032_64),
            Unwrite(loss),
        )
        self.wait(1)


class DPAdam00032RawSharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_raw_sharpness_axes(
            [
                "lr_0.00032_opt_adam",
                "lr_0.00032_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_0.00032_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_0.00032_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        self.play(Write(ax))

        raw_sharpness_00032 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.00032_opt_adam"],
            color=BLUE,
            fill_opacity=0.5,
        )

        raw_sharpness_00032_16 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.00032_opt_adam_eps_16_mgn_3.0_bc_False"],
            color=GREEN,
        )
        raw_eps_label_00032_16 = self.get_eps_label(
            ax,
            self.raw_sharpness_logs["lr_0.00032_opt_adam_eps_16_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 16",
            color=GREEN,
        )

        raw_sharpness_00032_32 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.00032_opt_adam_eps_32_mgn_3.0_bc_False"],
            color=GOLD,
        )
        raw_eps_label_00032_32 = self.get_eps_label(
            ax,
            self.raw_sharpness_logs["lr_0.00032_opt_adam_eps_32_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 32",
            color=GOLD,
        )

        raw_sharpness_00032_64 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.00032_opt_adam_eps_64_mgn_3.0_bc_False"],
            color=MAROON,
        )
        raw_eps_label_00032_64 = self.get_eps_label(
            ax,
            self.raw_sharpness_logs["lr_0.00032_opt_adam_eps_64_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 64",
            color=MAROON,
        )

        loss = self.get_loss_plot(
            [
                "lr_0.00032_opt_adam",
                "lr_0.00032_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_0.00032_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_0.00032_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        loss.set_z_index(0)

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Raw Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(r"DP-Adam Raw Sharpness ($\eta=3.2\text{e-}4$)", font_size=24)
        title.to_edge(UP)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(raw_sharpness_00032),
            Write(raw_sharpness_00032_16),
            Write(raw_eps_label_00032_16),
            Write(raw_sharpness_00032_32),
            Write(raw_eps_label_00032_32),
            Write(raw_sharpness_00032_64),
            Write(raw_eps_label_00032_64),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(raw_sharpness_00032),
            Unwrite(raw_sharpness_00032_16),
            Unwrite(raw_eps_label_00032_16),
            Unwrite(raw_sharpness_00032_32),
            Unwrite(raw_eps_label_00032_32),
            Unwrite(raw_sharpness_00032_64),
            Unwrite(raw_eps_label_00032_64),
            Unwrite(loss),
        )
        self.wait(1)


class DPAdam0001Sharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_precond_sharpness_axes(
            [
                "lr_0.0001_opt_adam",
                "lr_0.0001_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_0.0001_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_0.0001_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        self.play(Write(ax))

        precond_sharpness_0001 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.0001_opt_adam"],
            color=BLUE,
            fill_opacity=0.5,
        )
        precond_threshold_0001 = self.get_precond_sharpness_threshold(
            ax,
            eta=0.0001,
            mathtex=r"\eta = 1\text{e-}4",
            color=BLUE,
        )

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Preconditioned Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(r"DP-Adam Precond Sharpness ($\eta=1\text{e-}4$)", font_size=24)
        title.to_edge(UP)

        precond_sharpness_0001_16 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.0001_opt_adam_eps_16_mgn_3.0_bc_False"],
            color=GREEN,
        )
        eps_label_0001_16 = self.get_eps_label(
            ax,
            self.precond_sharpness_logs["lr_0.0001_opt_adam_eps_16_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 16",
            color=GREEN,
        )

        precond_sharpness_0001_32 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.0001_opt_adam_eps_32_mgn_3.0_bc_False"],
            color=GOLD,
        )
        eps_label_0001_32 = self.get_eps_label(
            ax,
            self.precond_sharpness_logs["lr_0.0001_opt_adam_eps_32_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 32",
            color=GOLD,
        )

        precond_sharpness_0001_64 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_0.0001_opt_adam_eps_64_mgn_3.0_bc_False"],
            color=MAROON,
        )
        eps_label_0001_64 = self.get_eps_label(
            ax,
            self.precond_sharpness_logs["lr_0.0001_opt_adam_eps_64_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 64",
            color=MAROON,
        )

        loss = self.get_loss_plot(
            [
                "lr_0.0001_opt_adam",
                "lr_0.0001_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_0.0001_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_0.0001_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        loss.set_z_index(0)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(precond_sharpness_0001),
            Write(precond_threshold_0001),
            Write(precond_sharpness_0001_16),
            Write(eps_label_0001_16),
            Write(precond_sharpness_0001_32),
            Write(eps_label_0001_32),
            Write(precond_sharpness_0001_64),
            Write(eps_label_0001_64),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(precond_sharpness_0001),
            Unwrite(precond_threshold_0001),
            Unwrite(precond_sharpness_0001_16),
            Unwrite(eps_label_0001_16),
            Unwrite(precond_sharpness_0001_32),
            Unwrite(eps_label_0001_32),
            Unwrite(precond_sharpness_0001_64),
            Unwrite(eps_label_0001_64),
            Unwrite(loss),
        )
        self.wait(1)


class DPAdam0001RawSharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_raw_sharpness_axes(
            [
                "lr_0.0001_opt_adam",
                "lr_0.0001_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_0.0001_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_0.0001_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        self.play(Write(ax))

        raw_sharpness_0001 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.0001_opt_adam"],
            color=BLUE,
            fill_opacity=0.5,
        )

        raw_sharpness_0001_16 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.0001_opt_adam_eps_16_mgn_3.0_bc_False"],
            color=GREEN,
        )
        raw_eps_label_0001_16 = self.get_eps_label(
            ax,
            self.raw_sharpness_logs["lr_0.0001_opt_adam_eps_16_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 16",
            color=GREEN,
        )

        raw_sharpness_0001_32 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.0001_opt_adam_eps_32_mgn_3.0_bc_False"],
            color=GOLD,
        )
        raw_eps_label_0001_32 = self.get_eps_label(
            ax,
            self.raw_sharpness_logs["lr_0.0001_opt_adam_eps_32_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 32",
            color=GOLD,
        )

        raw_sharpness_0001_64 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_0.0001_opt_adam_eps_64_mgn_3.0_bc_False"],
            color=MAROON,
        )
        raw_eps_label_0001_64 = self.get_eps_label(
            ax,
            self.raw_sharpness_logs["lr_0.0001_opt_adam_eps_64_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 64",
            color=MAROON,
        )

        loss = self.get_loss_plot(
            [
                "lr_0.0001_opt_adam",
                "lr_0.0001_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_0.0001_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_0.0001_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        loss.set_z_index(0)

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Raw Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(r"DP-Adam Raw Sharpness ($\eta=1\text{e-}4$)", font_size=24)
        title.to_edge(UP)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(raw_sharpness_0001),
            Write(raw_sharpness_0001_16),
            Write(raw_eps_label_0001_16),
            Write(raw_sharpness_0001_32),
            Write(raw_eps_label_0001_32),
            Write(raw_sharpness_0001_64),
            Write(raw_eps_label_0001_64),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(raw_sharpness_0001),
            Unwrite(raw_sharpness_0001_16),
            Unwrite(raw_eps_label_0001_16),
            Unwrite(raw_sharpness_0001_32),
            Unwrite(raw_eps_label_0001_32),
            Unwrite(raw_sharpness_0001_64),
            Unwrite(raw_eps_label_0001_64),
            Unwrite(loss),
        )
        self.wait(1)


class DPAdam00003Sharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_precond_sharpness_axes(
            [
                "lr_3e-05_opt_adam",
                "lr_3e-05_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_3e-05_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_3e-05_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        self.play(Write(ax))

        precond_sharpness_00003 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_3e-05_opt_adam"],
            color=BLUE,
            fill_opacity=0.5,
        )
        precond_threshold_00003 = self.get_precond_sharpness_threshold(
            ax,
            eta=3e-05,
            mathtex=r"\eta = 3\text{e-}5",
            color=BLUE,
        )

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Preconditioned Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(r"DP-Adam Precond Sharpness ($\eta=3\text{e-}5$)", font_size=24)
        title.to_edge(UP)

        precond_sharpness_00003_16 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_3e-05_opt_adam_eps_16_mgn_3.0_bc_False"],
            color=GREEN,
        )
        eps_label_00003_16 = self.get_eps_label(
            ax,
            self.precond_sharpness_logs["lr_3e-05_opt_adam_eps_16_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 16",
            color=GREEN,
        )

        precond_sharpness_00003_32 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_3e-05_opt_adam_eps_32_mgn_3.0_bc_False"],
            color=GOLD,
        )
        eps_label_00003_32 = self.get_eps_label(
            ax,
            self.precond_sharpness_logs["lr_3e-05_opt_adam_eps_32_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 32",
            color=GOLD,
        )

        precond_sharpness_00003_64 = self.get_precond_sharpness_dots(
            ax,
            self.precond_sharpness_logs["lr_3e-05_opt_adam_eps_64_mgn_3.0_bc_False"],
            color=MAROON,
        )
        eps_label_00003_64 = self.get_eps_label(
            ax,
            self.precond_sharpness_logs["lr_3e-05_opt_adam_eps_64_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 64",
            color=MAROON,
        )

        loss = self.get_loss_plot(
            [
                "lr_3e-05_opt_adam",
                "lr_3e-05_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_3e-05_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_3e-05_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        loss.set_z_index(0)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(precond_sharpness_00003),
            Write(precond_threshold_00003),
            Write(precond_sharpness_00003_16),
            Write(eps_label_00003_16),
            Write(precond_sharpness_00003_32),
            Write(eps_label_00003_32),
            Write(precond_sharpness_00003_64),
            Write(eps_label_00003_64),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(precond_sharpness_00003),
            Unwrite(precond_threshold_00003),
            Unwrite(precond_sharpness_00003_16),
            Unwrite(eps_label_00003_16),
            Unwrite(precond_sharpness_00003_32),
            Unwrite(eps_label_00003_32),
            Unwrite(precond_sharpness_00003_64),
            Unwrite(eps_label_00003_64),
            Unwrite(loss),
        )
        self.wait(1)


class DPAdam00003RawSharpness(SharpnessScene):
    def construct(self):
        self.next_section()
        ax = self.get_raw_sharpness_axes(
            [
                "lr_3e-05_opt_adam",
                "lr_3e-05_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_3e-05_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_3e-05_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        self.play(Write(ax))

        raw_sharpness_00003 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_3e-05_opt_adam"],
            color=BLUE,
            fill_opacity=0.5,
        )

        raw_sharpness_00003_16 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_3e-05_opt_adam_eps_16_mgn_3.0_bc_False"],
            color=GREEN,
        )
        raw_eps_label_00003_16 = self.get_eps_label(
            ax,
            self.raw_sharpness_logs["lr_3e-05_opt_adam_eps_16_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 16",
            color=GREEN,
        )

        raw_sharpness_00003_32 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_3e-05_opt_adam_eps_32_mgn_3.0_bc_False"],
            color=GOLD,
        )
        raw_eps_label_00003_32 = self.get_eps_label(
            ax,
            self.raw_sharpness_logs["lr_3e-05_opt_adam_eps_32_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 32",
            color=GOLD,
        )

        raw_sharpness_00003_64 = self.get_sharpness_dots(
            ax,
            self.raw_sharpness_logs["lr_3e-05_opt_adam_eps_64_mgn_3.0_bc_False"],
            color=MAROON,
        )
        raw_eps_label_00003_64 = self.get_eps_label(
            ax,
            self.raw_sharpness_logs["lr_3e-05_opt_adam_eps_64_mgn_3.0_bc_False"],
            mathtex=r"\varepsilon = 64",
            color=MAROON,
        )

        loss = self.get_loss_plot(
            [
                "lr_3e-05_opt_adam",
                "lr_3e-05_opt_adam_eps_16_mgn_3.0_bc_False",
                "lr_3e-05_opt_adam_eps_32_mgn_3.0_bc_False",
                "lr_3e-05_opt_adam_eps_64_mgn_3.0_bc_False",
            ]
        )
        loss.set_z_index(0)

        x_label = Tex(r"Training Steps", font_size=16)
        x_label.to_edge(DOWN, buff=SMALL_BUFF)

        y_label = Tex(r"Raw Sharpness", font_size=16)
        y_label.rotate(90 * DEGREES)
        y_label.to_edge(LEFT, buff=SMALL_BUFF)

        title = Tex(r"DP-Adam Raw Sharpness ($\eta=3\text{e-}5$)", font_size=24)
        title.to_edge(UP)

        self.play(
            Write(x_label),
            Write(y_label),
            Write(title),
            Write(raw_sharpness_00003),
            Write(raw_sharpness_00003_16),
            Write(raw_eps_label_00003_16),
            Write(raw_sharpness_00003_32),
            Write(raw_eps_label_00003_32),
            Write(raw_sharpness_00003_64),
            Write(raw_eps_label_00003_64),
            Write(loss),
        )
        self.wait(1)

        self.next_section()
        self.play(
            Unwrite(ax),
            Unwrite(x_label),
            Unwrite(y_label),
            Unwrite(title),
            Unwrite(raw_sharpness_00003),
            Unwrite(raw_sharpness_00003_16),
            Unwrite(raw_eps_label_00003_16),
            Unwrite(raw_sharpness_00003_32),
            Unwrite(raw_eps_label_00003_32),
            Unwrite(raw_sharpness_00003_64),
            Unwrite(raw_eps_label_00003_64),
            Unwrite(loss),
        )
        self.wait(1)
