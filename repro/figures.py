"""Figures 1-8 (190 mm; PDF/SVG/TIFF 500 dpi) from the regenerated results; written to figures_scripts/output/.
Run after table2, table3_4 and table5."""
import runpy, os
from _common import ROOT
os.chdir(ROOT / 'figures_scripts')
import sys; sys.path.insert(0, str(ROOT / 'figures_scripts'))
runpy.run_path(str(ROOT / 'figures_scripts' / 'gerar_restantes.py'), run_name='__main__')
