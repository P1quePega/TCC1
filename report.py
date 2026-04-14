"""
report.py — Gerador de relatório HTML autocontido — AeroStruct Suite
"""
import os
import base64
from datetime import datetime


def _img_to_b64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext  = os.path.splitext(path)[1].lower().strip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    return f"data:{mime};base64,{data}"


def _logo_tag():
    for name in ("logo.png", "logo.jpg", "logo.jpeg"):
        p = os.path.join(os.path.dirname(__file__), name)
        if os.path.exists(p):
            b64 = _img_to_b64(p)
            if b64:
                return f'<img src="{b64}" alt="Logo" style="height:30px;vertical-align:middle;" />'
    return '<span style="font-size:20px;">✈</span>'


def _find_images(img_dir):
    imgs = {}
    if not os.path.isdir(img_dir):
        return imgs
    mapping = {
        "01_geometria": "geometria", "02_malha": "malha", "03_bc": "bc",
        "04_stress": "stress", "05_deslocamento": "desl", "06_topologia": "topo",
    }
    for f in os.listdir(img_dir):
        if not f.lower().endswith(".png"):
            continue
        full = os.path.join(img_dir, f)
        low  = f.lower()
        for prefix, key in mapping.items():
            if low.startswith(prefix):
                imgs[key] = full; break
        else:
            if   "stress"  in low:                    imgs.setdefault("stress",    full)
            elif "desl"    in low or "disp"  in low:  imgs.setdefault("desl",      full)
            elif "topo"    in low or "density" in low: imgs.setdefault("topo",     full)
            elif "malha"   in low or "mesh"  in low:  imgs.setdefault("malha",     full)
            elif "geom"    in low:                    imgs.setdefault("geometria",  full)
            elif "bc"      in low or "contorno" in low: imgs.setdefault("bc",      full)
    return imgs


def _img_card(num, titulo, key, imgs):
    if key not in imgs:
        body = f'<div class="ph">Imagem não disponível<br><span>{titulo}</span></div>'
    else:
        b64 = _img_to_b64(imgs[key])
        body = f'<img src="{b64}" alt="{titulo}" />' if b64 else '<div class="ph">Erro ao carregar</div>'
    return f'''
    <div class="img-card">
      <div class="img-hdr"><span class="num">{num}</span><span class="ttl">{titulo}</span></div>
      <div class="img-body">{body}</div>
    </div>'''


def _row(label, val, hl=False):
    cls = ' class="hl"' if hl else ""
    return f"<tr{cls}><td>{label}</td><td>{val}</td></tr>"


def _section(title):
    return f'''
    <div class="sec-hdr">
      <div class="sec-dot"></div>
      <div class="sec-txt">{title.upper()}</div>
      <div class="sec-line"></div>
    </div>'''


def gerar_relatorio_html(results, output_path):
    p   = results["params"]
    now = datetime.fromisoformat(results["timestamp"]).strftime("%d/%m/%Y — %H:%M:%S")
    imgs = _find_images(results.get("img_dir", ""))

    tensao_ok  = results["stress_max"] <= p["tensao_max"]
    bnr_cls    = "ok" if tensao_ok else "fail"
    bnr_icon   = "✓" if tensao_ok else "✗"
    bnr_title  = "ANÁLISE APROVADA" if tensao_ok else "ANÁLISE REPROVADA"
    bnr_desc   = ("Tensão de Von Mises dentro do limite admissível."
                  if tensao_ok else "Tensão Von Mises excedeu o limite — revisão necessária.")
    status_val = (
        f'<span class="ok-txt">✓ APROVADO — {results["stress_max"]:.3f} ≤ {p["tensao_max"]} MPa</span>'
        if tensao_ok else
        f'<span class="fail-txt">✗ REPROVADO — {results["stress_max"]:.3f} > {p["tensao_max"]} MPa</span>'
    )
    fonte_perfil = (f"NACA {p.get('naca_code','----')}"
                    if p.get("fonte_perfil") == "naca" else p.get("caminho_dat", "N/A"))

    pre_imgs  = "".join([
        _img_card("01", "Geometria — Áreas do Modelo", "geometria", imgs),
        _img_card("02", "Malha de Elementos Finitos",  "malha",     imgs),
        _img_card("03", "Condições de Contorno",        "bc",        imgs),
    ])
    post_imgs = "".join([
        _img_card("04", "Tensão de Von Mises",              "stress", imgs),
        _img_card("05", "Deslocamento Total",               "desl",   imgs),
        _img_card("06", "Resultado Topológico (Densidade)", "topo",   imgs),
    ])

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Relatório — {p['nome_projeto']}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#111318;color:#C8D4EC;font-size:13px;line-height:1.5}}
.wrap{{max-width:1100px;margin:0 auto;padding-bottom:48px}}
.topbar{{display:flex;align-items:center;gap:12px;padding:13px 24px;background:#111318;border-bottom:1px solid #30395A}}
.app-name{{font-size:15px;font-weight:700;color:#C8D4EC;letter-spacing:.03em}}
.app-ver{{font-size:9px;color:#6A7A9C;background:#282E48;border:1px solid #30395A;border-radius:3px;padding:2px 7px;margin-left:4px}}
.topbar-right{{margin-left:auto;font-size:10px;color:#3A4562}}
.mod-hdr{{display:flex;align-items:center;padding:18px 24px;background:#21263C;border-bottom:1px solid #30395A;margin-bottom:24px}}
.mod-crumb{{font-size:9px;color:#3A4562;letter-spacing:.12em;text-transform:uppercase;margin-bottom:4px}}
.mod-title{{font-size:18px;font-weight:700;color:#C8D4EC}}
.mod-ts{{margin-left:auto;text-align:right;font-size:10px;color:#3A4562;line-height:1.8}}
.pad{{padding:0 24px}}
.banner{{display:flex;align-items:center;gap:14px;padding:14px 18px;border-radius:5px;border:1px solid;margin-bottom:22px}}
.banner.ok{{background:rgba(78,200,138,.07);border-color:rgba(78,200,138,.25)}}
.banner.fail{{background:rgba(217,82,82,.07);border-color:rgba(217,82,82,.25)}}
.b-icon{{font-size:20px;flex-shrink:0}}
.b-title{{font-size:14px;font-weight:700;letter-spacing:.06em}}
.banner.ok .b-title{{color:#4EC88A}}
.banner.fail .b-title{{color:#D95252}}
.b-desc{{font-size:11px;color:#6A7A9C;margin-top:2px}}
.b-meta{{margin-left:auto;text-align:right;font-size:10px;color:#3A4562;line-height:1.8}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:10px;margin-bottom:26px}}
.kpi{{background:#21263C;border:1px solid #30395A;border-radius:5px;padding:13px 15px;border-top:2px solid #4D82D6}}
.kpi.ok-k{{border-top-color:#4EC88A}}.kpi.fail-k{{border-top-color:#D95252}}.kpi.warn-k{{border-top-color:#D9963A}}
.kpi-val{{font-family:'Consolas',monospace;font-size:20px;font-weight:700;color:#4D82D6;line-height:1;margin-bottom:5px}}
.kpi.ok-k .kpi-val{{color:#4EC88A}}.kpi.fail-k .kpi-val{{color:#D95252}}.kpi.warn-k .kpi-val{{color:#D9963A}}
.kpi-lbl{{font-size:9px;color:#6A7A9C;letter-spacing:.05em;text-transform:uppercase;font-weight:600}}
.kpi-unit{{font-size:9px;color:#3A4562;margin-top:2px}}
.sec-hdr{{display:flex;align-items:center;gap:8px;margin-bottom:12px;margin-top:6px}}
.sec-dot{{width:5px;height:5px;border-radius:50%;background:#4D82D6;flex-shrink:0}}
.sec-txt{{font-size:10px;font-weight:700;letter-spacing:.14em;color:#6A7A9C;text-transform:uppercase;white-space:nowrap}}
.sec-line{{flex:1;height:1px;background:#30395A}}
.tbl{{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:22px}}
.tbl th{{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:#3A4562;padding:7px 12px;text-align:left;border-bottom:1px solid #30395A;font-weight:600}}
.tbl td{{padding:8px 12px;border-bottom:1px solid rgba(48,57,90,.55);color:#C8D4EC}}
.tbl tr:hover td{{background:rgba(33,38,60,.6)}}
.tbl td:last-child{{font-family:'Consolas',monospace;font-size:11px;color:#4D82D6;font-weight:600}}
.tbl .hl td{{background:rgba(77,130,214,.06)}}.tbl .hl td:last-child{{color:#4EC88A}}
.ok-txt{{color:#4EC88A;font-weight:700}}.fail-txt{{color:#D95252;font-weight:700}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:22px}}
@media(max-width:750px){{.two-col{{grid-template-columns:1fr}}}}
.tbl-blk-title{{font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6A7A9C;margin-bottom:7px;padding-bottom:5px;border-bottom:1px solid #30395A}}
.img-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px;margin-bottom:26px}}
.img-card{{background:#21263C;border:1px solid #30395A;border-radius:5px;overflow:hidden}}
.img-hdr{{display:flex;align-items:center;gap:8px;padding:9px 12px;border-bottom:1px solid #30395A;background:#282E48}}
.num{{font-family:'Consolas',monospace;font-size:9px;background:#1C2030;color:#4D82D6;border:1px solid #3E4C70;padding:1px 6px;border-radius:2px;font-weight:700;letter-spacing:.04em}}
.ttl{{font-size:11px;font-weight:500;color:#6A7A9C}}
.img-body{{padding:10px;min-height:160px;display:flex;align-items:center;justify-content:center}}
.img-body img{{max-width:100%;border-radius:3px}}
.ph{{font-size:11px;color:#3A4562;text-align:center;line-height:1.9}}
.ph span{{display:block;font-size:9px;color:#30395A;margin-top:2px}}
.rpt-footer{{display:flex;align-items:center;gap:10px;padding:16px 24px;border-top:1px solid #30395A;margin-top:10px;font-size:9px;color:#3A4562;letter-spacing:.08em}}
.rpt-footer .sep{{color:#3E4C70}}
</style>
</head>
<body>
<div class="wrap">
<header class="topbar">
  {_logo_tag()}
  <div><span class="app-name">AeroStruct Suite</span><span class="app-ver">v2.1</span></div>
  <div class="topbar-right">RELATÓRIO EXPORTADO &nbsp;|&nbsp; {now}</div>
</header>
<div class="mod-hdr">
  <div>
    <div class="mod-crumb">MÓDULOS / ESTRUTURAL / NERVURA / RELATÓRIO</div>
    <div class="mod-title">Otimização Topológica de Nervura</div>
  </div>
  <div class="mod-ts"><div>PROJETO: {p['nome_projeto'].upper()}</div><div>{now}</div></div>
</div>
<div class="pad">
  <div class="banner {bnr_cls}">
    <div class="b-icon">{bnr_icon}</div>
    <div><div class="b-title">{bnr_title}</div><div class="b-desc">{bnr_desc}</div></div>
    <div class="b-meta"><div>PROJETO: {p['nome_projeto'].upper()}</div><div>{now}</div></div>
  </div>
  <div class="kpi-grid">
    <div class="kpi {'ok-k' if tensao_ok else 'fail-k'}">
      <div class="kpi-val">{results['stress_max']:.3f}</div>
      <div class="kpi-lbl">Von Mises Máx.</div><div class="kpi-unit">MPa</div>
    </div>
    <div class="kpi"><div class="kpi-val">{results['stress_mean']:.3f}</div>
      <div class="kpi-lbl">Von Mises Médio</div><div class="kpi-unit">MPa</div></div>
    <div class="kpi"><div class="kpi-val">{results['disp_max']:.4f}</div>
      <div class="kpi-lbl">Deslocamento Máx.</div><div class="kpi-unit">mm</div></div>
    <div class="kpi warn-k"><div class="kpi-val">{results['n_elementos']:,}</div>
      <div class="kpi-lbl">Elementos</div><div class="kpi-unit">MALHA MEF</div></div>
    <div class="kpi"><div class="kpi-val">{results['n_nos']:,}</div>
      <div class="kpi-lbl">Nós</div><div class="kpi-unit">MALHA MEF</div></div>
    <div class="kpi"><div class="kpi-val">{results['area_casca']:.1f}</div>
      <div class="kpi-lbl">Área da Casca</div><div class="kpi-unit">mm²</div></div>
  </div>
  {_section("Verificação da Restrição de Tensão")}
  <table class="tbl"><tr><th>Critério</th><th>Valor</th></tr>
    {_row("Tensão máxima admissível", f"{p['tensao_max']} MPa")}
    {_row("Tensão Von Mises máxima obtida", f"{results['stress_max']:.4f} MPa")}
    {_row("Status", status_val, hl=True)}
  </table>
  {_section("Pré-Processamento — Geometria, Malha e Condições de Contorno")}
  <div class="img-grid">{pre_imgs}</div>
  {_section("Resultados — Tensão, Deslocamento e Topologia")}
  <div class="img-grid">{post_imgs}</div>
  {_section("Parâmetros da Análise")}
  <div class="two-col">
    <div><div class="tbl-blk-title">Modelo &amp; Geometria</div>
    <table class="tbl"><tr><th>Parâmetro</th><th>Valor</th></tr>
      {_row("Fonte do perfil", fonte_perfil)}
      {_row("Corda", f"{p['corda_mm']} mm")}
      {_row("Espessura da casca", f"{p['espessura_casca']} mm")}
      {_row("Engaste (ini / fim)", f"{p['x_long_ini_pct']*100:.1f}% / {p['x_long_fim_pct']*100:.1f}%")}
      {_row("Tamanho do elemento", f"{p['tamanho_elemento']} mm")}
    </table></div>
    <div><div class="tbl-blk-title">Material &amp; Cargas</div>
    <table class="tbl"><tr><th>Parâmetro</th><th>Valor</th></tr>
      {_row("Pressão aerodinâmica", f"{p['pressao_aerodinamica']} MPa")}
      {_row("Módulo de elasticidade (Ex)", f"{p['modulo_elasticidade']} MPa")}
      {_row("Coef. de Poisson", str(p['poisson']))}
      {_row("Densidade", f"{p['densidade']} t/mm³")}
    </table></div>
  </div>
  {_section("Otimização Topológica")}
  <table class="tbl"><tr><th>Parâmetro</th><th>Valor</th></tr>
    {_row("Objetivo", "Minimizar massa com restrição de tensão (SIMP)")}
    {_row("Região otimizável", "COMP_YELLOW (interior da nervura)")}
    {_row("Tensão máxima (restrição)", f"{p['tensao_max']} MPa")}
    {_row("Máx. iterações", str(p['max_iter']))}
    {_row("Convergência", str(p['convergencia']))}
  </table>
</div>
<footer class="rpt-footer">
  <span>AEROSTRUCT SUITE</span><span class="sep">|</span>
  <span>PyMAPDL + Shapely + SIMP</span><span class="sep">|</span>
  <span>PIPELINE v2.1</span><span class="sep">|</span>
  <span>{now}</span>
</footer>
</div></body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)