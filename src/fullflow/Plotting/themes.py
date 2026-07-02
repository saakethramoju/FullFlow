from __future__ import annotations


DARK_COLORS = [
    "#ffff00",  # yellow
    "#00ffff",  # cyan
    "#00ff00",  # lime
    "#ff00ff",  # magenta
    "#ff9900",  # orange
    "#ff3333",  # red
    "#33aaff",  # blue
    "#ffffff",  # white
    "#cc66ff",  # violet
    "#66ff99",  # spring green
]

LIGHT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
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
            "text_color": "#f2f2f2",
            "spine_color": "#d0d0d0",
            "grid_color": "#404040",
            "legend_facecolor": "#111111",
            "legend_edgecolor": "#808080",
        }

    return {
        "figure_facecolor": "#ffffff",
        "axes_facecolor": "#ffffff",
        "text_color": "#111111",
        "spine_color": "#222222",
        "grid_color": "#d0d0d0",
        "legend_facecolor": "#ffffff",
        "legend_edgecolor": "#808080",
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

        ax.tick_params(colors=settings["text_color"])

        ax.xaxis.label.set_color(settings["text_color"])
        ax.yaxis.label.set_color(settings["text_color"])
        ax.title.set_color(settings["text_color"])

        for spine in ax.spines.values():
            spine.set_color(settings["spine_color"])


def grid_kwargs(theme: str):
    settings = theme_settings(theme)

    return {
        "color": settings["grid_color"],
        "linestyle": "-",
        "linewidth": 0.6,
        "alpha": 0.8,
    }


def style_legend(legend, theme: str):
    if legend is None:
        return

    settings = theme_settings(theme)

    frame = legend.get_frame()
    frame.set_facecolor(settings["legend_facecolor"])
    frame.set_edgecolor(settings["legend_edgecolor"])
    frame.set_alpha(0.9)

    for text in legend.get_texts():
        text.set_color(settings["text_color"])


def style_colorbar(colorbar, theme: str):
    settings = theme_settings(theme)

    colorbar.ax.yaxis.label.set_color(settings["text_color"])
    colorbar.ax.tick_params(colors=settings["text_color"])

    for spine in colorbar.ax.spines.values():
        spine.set_color(settings["spine_color"])