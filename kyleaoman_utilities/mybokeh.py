import os
import bokeh.plotting

mytheme = bokeh.themes.Theme(os.path.expanduser("~/.bokehtheme.yaml"))


def custom_figure(figfunc):
    def wrapped(*args, **kwargs):

        # set default tools
        default_tools = [
            bokeh.models.tools.PanTool(),
            bokeh.models.tools.BoxZoomTool(),
            bokeh.models.tools.WheelZoomTool(),
            bokeh.models.tools.SaveTool(),
            bokeh.models.tools.ResetTool(),
        ]
        if "tools" not in kwargs:
            kwargs["tools"] = default_tools

        # initialize
        fig = figfunc(*args, **kwargs)

        # quad axes
        right_axis = type(fig.yaxis[0])()
        top_axis = type(fig.xaxis[0])()
        right_axis.ticker._property_values = fig.yaxis.ticker._property_values
        top_axis.ticker._property_values = fig.xaxis.ticker._property_values
        right_axis.major_label_text_alpha = 0
        top_axis.major_label_text_alpha = 0
        fig.add_layout(right_axis, "right")
        fig.add_layout(top_axis, "above")
        return fig

    return wrapped
