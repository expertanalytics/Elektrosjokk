import xalbrain as xb


def get_stimulus():
    stimulus = xb.Expression(
        "(2 < t && t < 5) ? 800*exp(-0.003*pow(x[0] - 0.5, 2))*exp(-0.003*pow(x[1] - 0.5, 2)) : 0.0",
        t=time,
        degree=1
    )
    return stimulus
