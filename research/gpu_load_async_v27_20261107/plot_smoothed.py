"""重绘office3轨迹图，仅平滑耗时展示。"""
from pathlib import Path
import hashlib

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import numpy as np

from analyze import COLORS, DUTIES, LABELS, METHODS, read, rolling_mean


def centered_mean(values, width=51):
    values=np.asarray(values,dtype=float)
    weights=np.ones(min(width,len(values)))
    return np.convolve(values,weights,mode='same')/np.convolve(np.ones_like(values),weights,mode='same')


def main():
    reports=Path(__file__).resolve().parent/'results/reports'
    source=reports/'TRAJECTORY_WITH_PSNR.csv'
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    rows=[row for row in read(source) if row['scene']=='office3']
    assert len(rows)==2000*len(METHODS)
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,3,figsize=(11.7,3.7))
        for method in METHODS:
            selected=sorted([row for row in rows if row['method']==method],key=lambda row:int(row['frame']))
            frames=[int(row['frame']) for row in selected]
            width=1.6 if method=='ours_adaptive' else 1.1
            axes[0].plot(frames,[int(row['K']) for row in selected],color=COLORS[method],lw=width,label=LABELS[method])
            for segment in range(8):
                part=[row for row in selected if int(row['segment'])==segment]
                x=[int(row['frame']) for row in part]
                elapsed=centered_mean([float(row['request_ms']) for row in part])
                assert len(elapsed)==len(part) and np.isfinite(elapsed).all()
                axes[1].plot(x,elapsed,color=COLORS[method],lw=width)
                axes[2].plot(x,rolling_mean([float(row['psnr']) for row in part],81),color=COLORS[method],lw=width)
        axes[0].set_yscale('log')
        axes[0].set_ylabel('Submitted Gaussians (log scale)')
        axes[1].set_yscale('log')
        axes[1].set_ylabel('End-to-end time (ms), log, smoothed')
        axes[1].yaxis.set_major_locator(LogLocator(base=2))
        axes[1].yaxis.set_major_formatter(ScalarFormatter())
        axes[1].yaxis.set_minor_formatter(NullFormatter())
        axes[2].set_ylabel('PSNR (dB), smoothed within stage')
        for axis in axes:
            axis.set_xlim(1,2000)
            axis.set_xlabel('Frame index')
            axis.grid(alpha=.18)
            for segment,stage in enumerate((0,1,2,3,3,2,1,0)):
                if segment:
                    axis.axvline(segment*250,color='#737373',ls='--',lw=.6,alpha=.6)
                axis.text((segment+.5)/8,1.025,f'{100*DUTIES[stage]:.0f}%',transform=axis.transAxes,ha='center',fontsize=8)
        fig.suptitle('Replica office3 — rising and falling controlled load',fontsize=11,y=.99)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.045),
                   ncol=5,frameon=False,fontsize=8)
        fig.text(.5,.012,'Latency: 51-frame centered mean within each load stage',ha='center',fontsize=7.5)
        fig.tight_layout(rect=[0,.18,1,.91])
        for suffix in ('png','pdf'):
            fig.savefig(reports/f'OFFICE3_TRAJECTORY_SMOOTHED.{suffix}',dpi=240)
        plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    print('SMOOTHED_OFFICE3_READY window=51 source_unchanged=true')


if __name__=='__main__':
    main()
