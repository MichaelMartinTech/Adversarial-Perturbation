from enum import Enum
import matplotlib.patches as mpatches


# Color coding for t-SNE
class Plot_Colors(str, Enum):
    CLEAN = 'xkcd:goldenrod'
    GLAZE = 'xkcd:lavender'
    SHADE = 'xkcd:green'
    NS_GL = 'xkcd:azure'


def is_shaded_glazed(file_name: str) -> str:
    return 'glazed' in file_name and 'shaded' in file_name

def is_glazed(file_name: str) -> str:
    return 'glazed' in file_name

def is_shaded(file_name: str) -> str:
    return 'shaded' in file_name

MPATCHES = [
    mpatches.Patch(color=Plot_Colors.CLEAN, label='Clean'),
    mpatches.Patch(color=Plot_Colors.GLAZE, label='Glazed'),
    mpatches.Patch(color=Plot_Colors.SHADE, label='Shaded'),
    mpatches.Patch(color=Plot_Colors.NS_GL, label='Shaded then Glazed')
]