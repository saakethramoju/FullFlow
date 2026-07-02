from __future__ import annotations


DARK_COLORS = [
    "#00ff00",  # green
    "#ff0000",  # red
    "#ffff00",  # yellow
    "#00ffff",  # cyan
    "#ff00ff",  # magenta
    "#ff9900",  # orange
    "#ffffff",  # white
    "#3399ff",  # blue
    "#ff66cc",  # pink
    "#99ff00",  # lime
]

LIGHT_COLORS = [
    "#b00000",  # dark red
    "#0047ab",  # stronger blue
    "#007a33",  # darker green
    "#a020a0",  # purple / magenta
    "#c46a00",  # burnt orange
    "#007a7a",  # teal
    "#5a3dbb",  # violet-blue
    "#000000",  # black
    "#8a6a00",  # olive-gold
    "#444444",  # dark gray
]


def check_theme(theme: str) -> str:
    theme = str(theme).lower().strip()

    if theme not in ("dark", "light"):
        raise ValueError("theme must be either 'dark' or 'light'.")

    return theme


def theme_colors(theme: str):
    theme = check_theme(theme)

    if theme == "dark":
        return DARK_COLORS

    return LIGHT_COLORS


def theme_settings(theme: str):
    theme = check_theme(theme)

    if theme == "dark":
        return {
            "figure_facecolor": "#000000",
            "axes_facecolor": "#000000",
            "text_color": "#ffffff",
            "spine_color": "#ffffff",
            "grid_color": "#8a8a8a",
            "legend_facecolor": "#000000",
            "legend_edgecolor": "#ffffff",
            "axis_color": "#ffffff",
            "line_width": 1.7,
            "spine_width": 1.0,
            "grid_linestyle": "--",
            "grid_linewidth": 0.7,
            "grid_alpha": 0.85,
            "font_family": "monospace",
            "label_weight": "bold",
            "map_cmap": "plasma",
        }

    return {
        "figure_facecolor": "#f7f7f2",
        "axes_facecolor": "#f7f7f2",
        "text_color": "#000000",
        "spine_color": "#000000",
        "grid_color": "#000000",
        "legend_facecolor": "#f7f7f2",
        "legend_edgecolor": "#000000",
        "axis_color": "#000000",
        "line_width": 1.7,
        "spine_width": 1.0,
        "grid_linestyle": "--",
        "grid_linewidth": 0.7,
        "grid_alpha": 0.60,
        "font_family": "monospace",
        "label_weight": "bold",
        "map_cmap": "viridis",
    }


def apply_theme(fig, axes, theme: str):
    settings = theme_settings(theme)
    colors = theme_colors(theme)

    if not isinstance(axes, (list, tuple)):
        axes = [axes]

    fig.patch.set_facecolor(settings["figure_facecolor"])

    for ax in axes:
        if ax is None:
            continue

        ax.set_facecolor(settings["axes_facecolor"])
        ax.set_prop_cycle(color=colors)

        ax.tick_params(
            axis="both",
            colors=settings["axis_color"],
            direction="in",
            length=5,
            width=1.0,
        )

        ax.xaxis.label.set_color(settings["axis_color"])
        ax.yaxis.label.set_color(settings["axis_color"])
        ax.title.set_color(settings["text_color"])

        ax.xaxis.label.set_fontfamily(settings["font_family"])
        ax.yaxis.label.set_fontfamily(settings["font_family"])
        ax.title.set_fontfamily(settings["font_family"])

        ax.xaxis.label.set_fontweight(settings["label_weight"])
        ax.yaxis.label.set_fontweight(settings["label_weight"])
        ax.title.set_fontweight(settings["label_weight"])

        for spine in ax.spines.values():
            spine.set_color(settings["spine_color"])
            spine.set_linewidth(settings["spine_width"])


def grid_kwargs(theme: str):
    settings = theme_settings(theme)

    return {
        "color": settings["grid_color"],
        "linestyle": settings["grid_linestyle"],
        "linewidth": settings["grid_linewidth"],
        "alpha": settings["grid_alpha"],
    }


def style_legend(legend, theme: str):
    if legend is None:
        return

    settings = theme_settings(theme)

    frame = legend.get_frame()
    frame.set_facecolor(settings["legend_facecolor"])
    frame.set_edgecolor(settings["legend_edgecolor"])
    frame.set_alpha(1.0)

    for text in legend.get_texts():
        text.set_color(settings["text_color"])
        text.set_fontfamily(settings["font_family"])


def style_colorbar(colorbar, theme: str):
    settings = theme_settings(theme)

    colorbar.ax.yaxis.label.set_color(settings["text_color"])
    colorbar.ax.yaxis.label.set_fontfamily(settings["font_family"])
    colorbar.ax.yaxis.label.set_fontweight(settings["label_weight"])

    colorbar.ax.tick_params(
        colors=settings["axis_color"],
        direction="in",
        length=5,
        width=1.0,
    )

    for spine in colorbar.ax.spines.values():
        spine.set_color(settings["spine_color"])
        spine.set_linewidth(settings["spine_width"])