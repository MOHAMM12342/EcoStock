"""Export du journal par tranche de 30 min vers Excel (openpyxl).

Respecte le schéma spécifié :
periode_30min, ca, prevision_mu, q10, q50, q90, reel, erreur, perte_potentielle, perte_reelle
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


HEADER_COLUMNS = [
    ("periode_30min", "Période 30 min", 22),
    ("ca", "CA (€)", 14),
    ("prevision_mu", "Prévision (µ)", 15),
    ("q10", "Quantile 10%", 14),
    ("q50", "Médiane (50%)", 14),
    ("q90", "Quantile 90%", 14),
    ("reel", "Réel (steaks)", 14),
    ("erreur", "Erreur (réel - prévu)", 20),
    ("perte_potentielle", "Perte potentielle", 18),
    ("perte_reelle", "Perte réelle (jetés)", 18),
]


def export_journal_to_excel(
    rows: List[Dict[str, Union[datetime, str, float, int]]],
    output_path: Union[str, Path],
) -> Path:
    """Exporte la liste des tranches de 30 minutes vers un classeur Excel soigné."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Journal Braise 30min"

    # Styles
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    fill_header = PatternFill(start_color="15100C", end_color="15100C", fill_type="solid")  # Noir charbon Braise
    align_center = Alignment(horizontal="center", vertical="center")
    align_right = Alignment(horizontal="right", vertical="center")
    border_thin = Border(
        left=Side(style="thin", color="D2C3A8"),
        right=Side(style="thin", color="D2C3A8"),
        top=Side(style="thin", color="D2C3A8"),
        bottom=Side(style="thin", color="D2C3A8"),
    )

    # Écriture des en-têtes
    for col_idx, (col_key, col_title, col_width) in enumerate(HEADER_COLUMNS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_title)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = align_center
        cell.border = border_thin
        ws.column_dimensions[get_column_letter(col_idx)].width = col_width

    # Écriture des données
    for row_idx, data_row in enumerate(rows, start=2):
        for col_idx, (col_key, _, _) in enumerate(HEADER_COLUMNS, start=1):
            val = data_row.get(col_key, 0)
            cell = ws.cell(row=row_idx, column=col_idx)

            if isinstance(val, datetime):
                cell.value = val.strftime("%Y-%m-%d %H:%M")
                cell.alignment = align_center
            else:
                cell.value = val
                cell.alignment = align_right

                # Formatage monétaire pour le CA
                if col_key == "ca":
                    cell.number_format = "#,##0.00 €"
                elif col_key in ("prevision_mu", "q10", "q50", "q90", "erreur"):
                    cell.number_format = "0.0"
                elif col_key in ("reel", "perte_potentielle", "perte_reelle"):
                    cell.number_format = "0"

            cell.border = border_thin

            # Alerte visuelle douce si perte réelle > 0
            if col_key == "perte_reelle" and isinstance(val, (int, float)) and val > 0:
                cell.fill = PatternFill(start_color="FDE8E8", end_color="FDE8E8", fill_type="solid")
                cell.font = Font(name="Calibri", color="9B1C1C", bold=True)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_file)
    return out_file

