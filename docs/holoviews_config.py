# https://www.holoviews.org/user_guide/Configuring.html

import holoviews as hv

hv.extension("bokeh")

# "Vibrant" color palette by Paul Tol, from https://personal.sron.nl/~pault/
palette = [
    "#0077BB",  # 0 blue
    "#33BBEE",  # 1 cyan
    "#009988",  # 2 teal
    "#EE7733",  # 3 orange
    "#CC3311",  # 4 red
    "#EE3377",  # 5 magenta
    #'#BBBBBB',  # grey
    "#888888",  # 6 slightly darker grey than the original
    "#000000",  # 7 black
]

# https://www.holoviews.org/user_guide/Applying_Customizations.html
hv.opts.defaults(
    hv.opts.Curve(
        color=hv.Cycle(palette)
    ),
    hv.opts.Scatter(
        color=hv.Cycle(palette)
    ),
    # TODO: how to set default width of all plots? (Curve, Scatter, Image, Layout ...)
    #width=400,
)
