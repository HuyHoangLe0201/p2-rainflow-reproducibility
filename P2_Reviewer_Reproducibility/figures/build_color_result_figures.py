"""Build the two inline vector result graphics from the archived P2 results."""
from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import math
import re

PALETTE = {
    "ink": "213247", "muted": "495C70", "axis": "6C7C8D", "grid": "DAE3EB",
    "blue": "2C5083", "amber": "A94F12", "teal": "007A73", "rose": "A3435B",
    "bluewash": "EBF1F9", "sand": "FAF1E4", "fog": "F2F4F7",
    "limit": "594666", "limitwash": "F6F1F7", "border": "A2B4C5",
}
SELECTORS = (
    ("hard_thermal", "Thermal baseline", "blue", "dot", "circle"),
    ("soft_RF", "Soft rainflow penalty", "amber", "dash", "triangle"),
    ("hard_RF", "Hard rainflow budget", "teal", "solid", "square"),
)
PATTERNS = {
    "dot": "dash pattern=on 0.85bp off 1.85bp",
    "dash": "dash pattern=on 4.15bp off 2.55bp",
    "solid": "solid",
}
REPORT = "output/research/P2_MissionRF_BoundaryScan_20261120/results.json"

def col(name):
    return "pTwo" + name

def norm(x):
    return f"{x:.4f}".rstrip("0").rstrip(".") if x else "0"

def pt(x, y):
    return f"({norm(x)},{norm(y)})"

class Tikz:
    def __init__(self, width, height):
        self.lines = [r"\begin{tikzpicture}[x=1bp,y=1bp,line cap=round,line join=round]"]
        self.lines += [rf"\definecolor{{{col(k)}}}{{HTML}}{{{v}}}" for k, v in PALETTE.items()]
        self.lines.append(rf"\path[use as bounding box] (0,0) rectangle {pt(width,height)};")

    def line(self, points, color, width=0.55, style=""):
        path = " -- ".join(pt(x,y) for x,y in points)
        opts = f"draw={col(color)},line width={width}bp"
        if style:
            opts += "," + style
        self.lines.append(rf"\draw[{opts}] {path};")

    def rect(self, x0, y0, x1, y1, fill="", stroke="", rounded=0, style=""):
        opts = []
        if fill:
            opts.append("fill=" + ("white" if fill=="white" else col(fill)))
        if stroke:
            opts.extend(("draw="+col(stroke),"line width=0.55bp"))
        if rounded:
            opts.append(f"rounded corners={rounded}bp")
        if style:
            opts.append(style)
        self.lines.append(rf"\path[{','.join(opts)}] {pt(x0,y0)} rectangle {pt(x1,y1)};")

    def node(self, x, y, label, *, size=8.5, anchor="center", color="ink", bold=False, rotate=None):
        weight = r"\bfseries" if bold else ""
        opts=rf"anchor={anchor},inner sep=0pt,text={col(color)},font={{\sffamily\fontsize{{{size}}}{{{size+1.2}}}\selectfont{weight}}}"
        if rotate:
            opts += f",rotate={rotate}"
        self.lines.append(rf"\node[{opts}] at {pt(x,y)} {{{label}}};")

    def marker(self, x, y, color, symbol, size=2.45):
        color_name = col(color)
        opts = f"draw={color_name},fill=white,line width=0.9bp"
        if symbol=="circle":
            self.lines.append(rf"\path[{opts}] {pt(x,y)} circle[radius={size}bp];")
        elif symbol=="square":
            self.lines.append(rf"\path[{opts}] {pt(x-size,y-size)} rectangle {pt(x+size,y+size)};")
        else:
            points = [pt(x,y+size*1.28), pt(x-size*1.09,y-size*.85), pt(x+size*1.09,y-size*.85)]
            self.lines.append(rf"\path[{opts}] {' -- '.join(points)} -- cycle;")

    def series(self, axis, xx, yy, color, width, style="", step=False):
        assert len(xx)==len(yy) and len(xx)>0
        coords=[]
        for i,(x,y) in enumerate(zip(xx,yy)):
            if step and i:
                coords.append(pt(axis.x(x),axis.y(yy[i-1])))
            coords.append(pt(axis.x(x),axis.y(y)))
        opts = f"draw={col(color)},line width={width}bp"
        if style:
            opts += "," + style
        self.lines.append(rf"\draw[{opts}] "+" --\n".join(coords)+";")

    def clipped(self, axis):
        self.lines.extend([r"\begin{scope}",
            rf"\clip {pt(axis.left,axis.bottom)} rectangle {pt(axis.right,axis.top)};"])

    def end_clip(self):
        self.lines.append(r"\end{scope}")

    def code(self):
        return "\n".join((*self.lines,r"\end{tikzpicture}"))

@dataclass
class Axes:
    left: float
    bottom: float
    right: float
    top: float
    x0: float
    x1: float
    y0: float
    y1: float

    def x(self, value):
        return self.left + (self.right-self.left)*(value-self.x0)/(self.x1-self.x0)

    def y(self, value):
        return self.bottom + (self.top-self.bottom)*(value-self.y0)/(self.y1-self.y0)

def tick_label(v):
    return rf"\({v:g}\)" if v<0 else f"{v:g}"

def axes(plot, a, x_ticks, y_ticks, *, xlabel="", ylabel="", small=False):
    sz=7.1 if small else 8.25
    for v in y_ticks:
        y=a.y(v)
        plot.line([(a.left,y),(a.right,y)],"grid",0.42)
        plot.line([(a.left-2.8,y),(a.left,y)],"axis",0.52)
        plot.node(a.left-5.25,y,tick_label(v),anchor="east",color="muted",size=sz)
    for v in x_ticks:
        x=a.x(v)
        plot.line([(x,a.bottom),(x,a.bottom-2.8)],"axis",0.52)
        plot.node(x,a.bottom-5.4,tick_label(v),anchor="north",color="muted",size=sz)
    plot.line([(a.left,a.top),(a.left,a.bottom),(a.right,a.bottom)],"axis",0.65)
    if xlabel:
        plot.node((a.left+a.right)/2,a.bottom-22,xlabel,anchor="north",size=8.8)
    if ylabel:
        plot.node(a.left-31,(a.bottom+a.top)/2,ylabel,rotate=90,size=8.8)

def heading(plot, a, panel, title):
    plot.node(a.left,a.top+19,f"({panel})",anchor="west",color="blue",size=11,bold=True)
    plot.node(a.left+21,a.top+19,title,anchor="west",size=10.7,bold=True)

def legend_item(plot, x, y, label, color, style, symbol):
    plot.line([(x,y),(x+23,y)],color,1.2,PATTERNS[style])
    plot.marker(x+11.5,y,color,symbol,2.2)
    plot.node(x+28,y,label,anchor="west",size=9)

def load(root):
    original=(root/REPORT).read_bytes()
    values=json.loads(original)
    data=json.loads((root/"figures/mission_curves.json").read_text("utf8"))
    assert sha256(original).hexdigest()==data["archived_report_sha256"]
    catalogue=values["full"][data["case"]]["controllers"]
    for key, *_ in SELECTORS:
        assert catalogue[key]["RF_max"]==data["selectors"][key]["terminal_charge"]
    changes=[new-old for old,new in zip(catalogue["hard_thermal"]["RF_per_device"],catalogue["hard_RF"]["RF_per_device"],strict=True)]
    assert len(changes)==30 and all(math.isfinite(v) and v for v in changes)
    assert sum(v<0 for v in changes)==14 and sum(v>0 for v in changes)==16
    assert data["budget"]==values["freeze"]["budget"]
    return data,changes

def device_plot(changes):
    p=Tikz(455,170)
    a=Axes(51,36,443,137,0.15,30.85,-.55,.76)
    axes(p,a,[1,5,10,15,20,25,30],[-.5,-.25,0,.25,.5,.75],
         xlabel="Device index",ylabel="Change in rainflow charge")
    p.line([(a.left,a.y(0)),(a.right,a.y(0))],"axis",0.82)
    for device,value in enumerate(changes,1):
        assert a.y0<=value<=a.y1
        x=a.x(device)
        color="teal" if value<0 else "rose"
        p.rect(x-3.6,min(a.y(0),a.y(value)),x+3.6,max(a.y(0),a.y(value)),fill=color)
        if abs(value)<0.008:
            p.lines.append(rf"\path[fill={col(color)}] {pt(x,a.y(value))} circle[radius=0.85bp];")
    for x,color,label in ((205,"teal","Lower charge (14)"),(326,"rose","Higher charge (16)")):
        p.rect(x,150,x+9,157,fill=color,rounded=1)
        p.node(x+15,153.5,label,anchor="west",size=9,color="muted")
    return p.code()

def phase_guides(plot, a, data):
    for v in data["phase_changes_s"]:
        x=a.x(v)
        plot.line([(x,a.bottom),(x,a.top)],"border",0.48,"dash pattern=on 1.8bp off 2.5bp")

def all_curves(plot, axis, xx, selectors, field, *, steps=False, inset=False):
    plot.clipped(axis)
    for key,_,color,style,_ in reversed(SELECTORS):
        yy=selectors[key][field]
        if field=="window_rms_A" and not inset:
            assert max(yy)<=axis.y1 and min(yy)>=axis.y0
        if field=="window_peak_bound_C" and not inset:
            assert max(yy)<=axis.y1 and min(yy)>=axis.y0
        plot.series(axis,xx,yy,color,(1.30 if key=="hard_RF" else 1.08),PATTERNS[style],step=steps)
    plot.end_clip()

def zoom(plot, parent, axes_box, centres, selectors, field, y_ticks):
    a=axes_box
    x0,x1,y0,y1=a.x0,a.x1,a.y0,a.y1
    selection=[i for i,x in enumerate(centres) if x0<=x<=x1]
    assert len(selection)==8
    bx0,by0,bx1,by1=parent.x(x0),parent.y(y0),parent.x(x1),parent.y(y1)
    plot.rect(bx0,by0,bx1,by1,stroke="axis",style="dash pattern=on 1.8bp off 1.5bp")
    plot.line([(bx0,by0),(a.right+11,a.top+8)],"border",0.46)
    xmin=parent.x0+(a.left-22-parent.left)*(parent.x1-parent.x0)/(parent.right-parent.left)
    xmax=parent.x0+(a.right+11-parent.left)*(parent.x1-parent.x0)/(parent.right-parent.left)
    covered=[parent.y(row[field][i]) for row in selectors.values()
             for i,x in enumerate(centres) if xmin-0.1<=x<=xmax+0.1]
    assert covered and (max(covered)<a.bottom-16 or min(covered)>a.top+11), "Inset would hide a main trace"
    plot.rect(a.left-22,a.bottom-16,a.right+11,a.top+11,fill="white",stroke="border",rounded=2.8)
    axes(plot,a,[25.8,26.2,26.6],y_ticks,small=True)
    plot.node(a.left+1,a.top+7.2,"Late braking",anchor="west",color="muted",size=7.55,bold=True)
    xx=[centres[i] for i in selection]
    series={key:{field:[selectors[key][field][i] for i in selection]} for key,*_ in SELECTORS}
    assert all(y0<=v<=y1 for row in series.values() for v in row[field])
    all_curves(plot,a,xx,series,field,inset=True)
    for j,(key,_,color,_,symbol) in enumerate(SELECTORS):
        for k in range(j,8,3):
            plot.marker(a.x(xx[k]),a.y(series[key][field][k]),color,symbol,1.52)

def terminal_plot(p, a, selectors, budget):
    limit=a.y(budget)
    p.rect(a.left,limit,a.right,a.top,fill="limitwash")
    p.line([(a.right,a.bottom),(a.right,a.top)],"axis",0.65)
    for v in (6.5,6.7,6.9):
        y=a.y(v)
        p.line([(a.left,y),(a.right,y)],"grid",0.42)
        p.line([(a.right,y),(a.right+2.5,y)],"axis",0.52)
        p.node(a.right+4.4,y,f"{v:g}",anchor="west",size=7.4,color="muted")
    p.line([(a.left,limit),(a.right,limit)],"limit",0.75,"dash pattern=on 3bp off 2bp")
    p.node((a.left+a.right)/2,limit+3.7,f"{budget:.3f}",anchor="south",size=7.45,color="limit")
    p.node((a.left+a.right)/2,a.top+5,"Terminal",anchor="south",size=8.3,bold=True)
    for i,(key,_,color,_,symbol) in enumerate(SELECTORS):
        value=selectors[key]["terminal_charge"]
        assert a.y0<value<a.y1
        x=a.left+(a.right-a.left)*(.19+.30*i)
        y=a.y(value)
        p.marker(x,y,color,symbol,2.7)
        p.node(x,y+(5 if i!=1 else -5),f"{value:.3f}",anchor="south" if i!=1 else "north",size=7.65,color=color,bold=True)

def mission_plot(data):
    p=Tikz(455,395)
    for x,(_,label,color,style,symbol) in zip((28,157,301),SELECTORS):
        legend_item(p,x,383,label,color,style,symbol)
    A=Axes(45,229,218,344,0,30,0,555)
    B=Axes(273,229,443,344,0,30,0,43)
    C=Axes(45,47,218,158,0,30,37,101)
    D=Axes(273,47,373,158,0,30,0,7.6)
    T=Axes(381,47,437,158,0,3,6.48,7.02)
    ticks=[0,5,10,15,20,25,30]
    phases=[0,*data["phase_changes_s"],30]
    for k,(label,shade) in enumerate((("Traction","bluewash"),("Brake","sand"),("Dwell","fog"))):
        x0,x1=A.x(phases[k]),A.x(phases[k+1])
        p.rect(x0,A.bottom,x1,A.top,fill=shade)
        p.node((x0+x1)/2,A.top-7.5,label,color="muted",size=7.3)
    axes(p,A,ticks,[0,200,400],ylabel="Reference amplitude (A)")
    phase_guides(p,A,data)
    heading(p,A,"a","Current demand")
    assert max(data["reference_crest_A"])<A.y1
    p.series(A,data["demand_time_s"],data["reference_crest_A"],"blue",1.20,step=True)

    edges=data["window_edges_s"]
    centres=[(a+b)/2 for a,b in zip(edges,edges[1:])]
    selectors=data["selectors"]
    axes(p,B,ticks,[0,10,20,30,40],ylabel="Windowed RMS error (A)")
    phase_guides(p,B,data)
    heading(p,B,"b","Current tracking")
    all_curves(p,B,centres,selectors,"window_rms_A")
    zoom(p,B,Axes(324,250,398,292,25.8,26.6,34,36.8),centres,selectors,"window_rms_A",[34,35,36])

    axes(p,C,ticks,[40,55,70,85,100],xlabel="Mission time (s)",ylabel=r"Peak bound (\({}^\circ{\rm C}\))")
    phase_guides(p,C,data)
    heading(p,C,"c","Thermal upper bound")
    all_curves(p,C,centres,selectors,"window_peak_bound_C")
    zoom(p,C,Axes(105,121,158,145,25.8,26.6,86.6,94.5),centres,selectors,"window_peak_bound_C",[87,90,93])

    axes(p,D,[0,10,20,30],[0,2,4,6],xlabel="Mission time (s)",ylabel="Maximum device charge")
    phase_guides(p,D,data)
    heading(p,D,"d","Rainflow charge")
    all_curves(p,D,edges,selectors,"settled_charge",steps=True)
    budget=data["budget"]
    y=D.y(budget)
    p.line([(D.left,y),(D.right,y)],"limit",0.75,"dash pattern=on 3bp off 2bp")
    p.node(D.left+4,y+3.7,"Budget",anchor="south west",size=7.8,color="limit")
    terminal_plot(p,T,selectors,budget)
    return p.code()

def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("supplement_root",type=Path)
    parser.add_argument("out_dir",type=Path)
    parser.add_argument("--verify-manuscript",type=Path)
    a=parser.parse_args()
    data,changes=load(a.supplement_root)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    results=(("figure3.tikz",device_plot(changes)),("figure4.tikz",mission_plot(data)))
    for name,content in results:
        (a.out_dir/name).write_text(content+"\n",encoding="utf8",newline="\n")
    if a.verify_manuscript:
        text=a.verify_manuscript.read_text("utf8")
        for name,content in results:
            assert text.count(content)==1, f"Embedded {name} does not match the archive"

if __name__=="__main__":
    main()

