"""Export the latest validated Markdown report and selected figures to a printable PDF."""
import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from neuralshift.utils.validation import validate_results


def export_pdf(results, output):
    root = Path(results)
    validate_results(root)
    reports = sorted(root.glob('reports/*/manuscript.md'))
    if not reports:
        raise ValueError('Generate the Markdown report first with the report CLI')
    source = reports[-1]
    output = Path(output)
    if output.exists():
        raise ValueError(f'Refusing to overwrite {output}; choose another output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyResearch', fontName='Helvetica', fontSize=9.5, leading=13, spaceAfter=7))
    styles['Title'].fontSize = 20
    styles['Title'].leading = 24
    story = []
    text = source.read_text(encoding='utf-8').split('## 19. Appendix')[0]
    in_table = False
    def plain(line):
        line = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', line)
        return escape(line.replace('**','').replace('`','').replace('\u2014','-').replace('\u2013','-'))
    for block in text.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        if block.startswith('|'):
            if not in_table:
                story.append(Paragraph('The complete per-run table is supplied in validated_metrics.csv. A compact clean-condition summary across independent seeds follows below.', styles['BodyResearch']))
            in_table = True
            continue
        if block.startswith('# '):
            story.append(Paragraph(plain(block[2:]), styles['Title']))
        elif block.startswith('## '):
            heading, _, body = block[3:].partition('\n')
            story.append(Paragraph(plain(heading), styles['Heading2']))
            if body.strip():
                story.append(Paragraph(plain(body).replace('\n', '<br/>'), styles['BodyResearch']))
        else:
            story.append(Paragraph(plain(block).replace('\n','<br/>'), styles['BodyResearch']))
    story.append(Paragraph('19. Appendix: verified tables and selected figures', styles['Heading2']))
    frame = pd.read_csv(source.parent/'seed_summary.csv')
    table = [['Clean metric','Calibration','Mean','SD','95% t interval','Seeds']]
    for metric in ['accuracy','nll','ece','auroc','aurc']:
        for cal in ['raw','temperature']:
            row = frame[(frame.corruption=='clean') & (frame.method=='msp') & (frame.metric==metric) & (frame.calibration==cal)].iloc[0]
            table.append([metric,cal,f'{row["mean"]:.4f}',f'{row["std"]:.4f}',f'[{row.ci_low:.4f}, {row.ci_high:.4f}]',str(int(row.n))])
    t = Table(table, colWidths=[72,78,53,53,125,40], repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#12334a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f8'),colors.white]),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    story.append(t)
    story.append(Spacer(1,12))
    story.append(Paragraph('Intervals describe variation between independent training seeds conditional on the fixed evaluated test subset. They do not quantify population sampling uncertainty. Full configurations, per-class metrics, predictions, and all figures are in the accompanying repository.',styles['BodyResearch']))
    runs = sorted(root.glob('runs/*'))
    representative = runs[0]
    latest_figures = sorted(representative.glob('figures/*'))[-1]
    names = ['01a_dataset_examples','02_training_convergence','03_corruption_severity','04_accuracy_degradation','05_confidence_reliability','06_calibration_clean_0','07_error_detection_clean_0','08_selective_prediction_clean_0','09_method_comparison']
    selected = [latest_figures/(name+'.png') for name in names]
    selected += sorted(root.glob('figures/statistics-*/10_*.png'))[-1:]
    for index,path in enumerate(selected,1):
        if not path.exists():
            raise ValueError(f'Missing selected figure {path}')
        story.append(PageBreak())
        story.append(Paragraph(f'Figure {index}: '+path.stem.split('_',1)[-1].replace('_',' '),styles['Heading2']))
        width,height = ImageReader(str(path)).getSize()
        scale = min(475/width,660/height)
        story.append(Image(str(path),width=width*scale,height=height*scale))
        story.append(Spacer(1,10))
        story.append(Paragraph('Source: '+plain(str(path.relative_to(root)))+'. Per-run panels identify the dataset, model, condition, and seed; the paired comparison uses all pilot seeds.',styles['BodyResearch']))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.HexColor('#607080'))
        canvas.drawString(42,25,'NeuralShift Lab | Empirical pilot study | Full research matrix pending')
        canvas.drawRightString(A4[0]-42,25,str(doc.page))
    SimpleDocTemplate(str(output),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=43,title='NeuralShift Lab - Predictive Reliability Pilot').build(story,onFirstPage=footer,onLaterPages=footer)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(export_pdf(args.results,args.output))
