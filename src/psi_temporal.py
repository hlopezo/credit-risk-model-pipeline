"""PSI mensual: aprendizaje exclusivo en referencia TRAIN, sin seleccion automatica."""
from pathlib import Path
import json, math, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
REFERENCE_MODE='early_train'
REFERENCE_N_MONTHS=3
REFERENCE_MIN_N=500
MIN_MONTH_OBS=200
MIN_EXPECTED_BIN_COUNT=5
MIN_TRAIN_MONTH_COVERAGE=.70
ALPHA=.5
N_BINS=5
AMBER=.10
RED=.20
AUTO_DROP_PSI=False
ROOT=Path('4.ANALIS_UV_BV_MV/PSI_TEMPORAL')

def configure(*, reference_mode='early_train', reference_n_months=3,
              reference_min_n=500, min_month_obs=200,
              min_expected_bin_count=5, min_train_month_coverage=.70,
              alpha=.5, auto_drop_psi=False):
    """Validate all settings before explicitly updating the existing module API."""
    global REFERENCE_MODE, REFERENCE_N_MONTHS, REFERENCE_MIN_N
    global MIN_MONTH_OBS, MIN_EXPECTED_BIN_COUNT, MIN_TRAIN_MONTH_COVERAGE
    global ALPHA, AUTO_DROP_PSI
    if reference_mode not in ('early_train', 'full_train'):
        raise ValueError('REFERENCE_MODE debe ser early_train o full_train')
    for name, value in [('REFERENCE_N_MONTHS', reference_n_months),
                        ('REFERENCE_MIN_N', reference_min_n),
                        ('MIN_MONTH_OBS', min_month_obs),
                        ('MIN_EXPECTED_BIN_COUNT', min_expected_bin_count)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} debe ser entero positivo')
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('ALPHA debe ser finito y positivo')
    if not np.isfinite(min_train_month_coverage) or not 0 < min_train_month_coverage <= 1:
        raise ValueError('Cobertura debe pertenecer a (0, 1]')
    if not isinstance(auto_drop_psi, bool):
        raise ValueError('AUTO_DROP_PSI debe ser booleano')
    REFERENCE_MODE = reference_mode
    REFERENCE_N_MONTHS = reference_n_months
    REFERENCE_MIN_N = reference_min_n
    MIN_MONTH_OBS = min_month_obs
    MIN_EXPECTED_BIN_COUNT = min_expected_bin_count
    MIN_TRAIN_MONTH_COVERAGE = min_train_month_coverage
    ALPHA = alpha
    AUTO_DROP_PSI = auto_drop_psi
    return get_configuration()


def get_configuration(reference_mode=None):
    """Snapshot parameters used in a calculation, independent of later settings."""
    return {
        'REFERENCE_MODE': REFERENCE_MODE if reference_mode is None else reference_mode,
        'REFERENCE_N_MONTHS': REFERENCE_N_MONTHS,
        'REFERENCE_MIN_N': REFERENCE_MIN_N,
        'MIN_MONTH_OBS': MIN_MONTH_OBS,
        'MIN_EXPECTED_BIN_COUNT': MIN_EXPECTED_BIN_COUNT,
        'MIN_TRAIN_MONTH_COVERAGE': MIN_TRAIN_MONTH_COVERAGE,
        'ALPHA': ALPHA, 'N_BINS': N_BINS, 'AMBER': AMBER, 'RED': RED,
        'AUTO_DROP_PSI': AUTO_DROP_PSI,
    }


def validate_samples(df, matrices):
    """Return sample/month diagnostics without modifying data or learning bins."""
    if not df.index.is_unique:
        raise ValueError('Indice original duplicado')
    temporal = periods(df.periodo_observacion)
    if 'codmes' in df and not periods(df.codmes).equals(temporal):
        raise ValueError('codmes no coincide con periodo_observacion')
    if set(matrices) != {'TRAIN', 'TEST', 'OOT'}:
        raise ValueError('Se requieren TRAIN, TEST y OOT')
    sample_rows, monthly_rows = [], []
    columns = matrices['TRAIN'].columns
    for segment, matrix in matrices.items():
        if not matrix.index.is_unique or not matrix.index.isin(df.index).all():
            raise ValueError('Indice invalido')
        if not matrix.columns.equals(columns):
            raise ValueError('Columnas distintas entre muestras')
        if not df.loc[matrix.index, 'SEGMENTO'].eq(segment).all():
            raise ValueError('Segmento desalineado')
        pd.testing.assert_frame_equal(matrix, df.loc[matrix.index, matrix.columns])
        month = temporal.loc[matrix.index]
        sample_rows.append({
            'Segmento': segment, 'N_registros': len(matrix),
            'N_variables': len(columns),
            'Primer_periodo': str(month.min()) if len(month) else None,
            'Ultimo_periodo': str(month.max()) if len(month) else None,
            'Indices_y_valores': 'VALIDADOS',
        })
        monthly_rows.extend({'Segmento': segment, 'Periodo': str(period),
                             'N_observaciones': int(count)}
                            for period, count in month.value_counts().sort_index().items())
    return pd.DataFrame(sample_rows), pd.DataFrame(monthly_rows)

def periods(s):
    t=s.astype('string').str.replace(r'\.0$','',regex=True)
    compact=t.str.fullmatch(r'\d{6}').fillna(False)
    result=pd.Series(pd.NaT,index=s.index,dtype='datetime64[ns]')
    result.loc[compact]=pd.to_datetime(t[compact],format='%Y%m',errors='raise')
    result.loc[~compact]=pd.to_datetime(t[~compact],errors='raise')
    if result.isna().any(): raise ValueError('Periodos nulos')
    return result.dt.to_period('M')

def token(v):
    return 'VALUE:'+type(v).__name__+':'+repr(v)

def learn_bins(reference,n_bins=None):
    n_bins=N_BINS if n_bins is None else n_bins
    s=pd.Series(reference); numeric=pd.api.types.is_numeric_dtype(s)
    finite=s[s.notna()]
    if numeric: finite=finite[np.isfinite(finite)]
    counts=finite.value_counts()
    cfg=dict(numeric=numeric,method='categories',categories=[],edges=[],dominant=None,
             state='OK',concentration=float(counts.iloc[0]/len(s)) if len(counts) and len(s) else np.nan)
    if not len(s) or not len(finite): cfg['state']='REFERENCIA_DEGENERADA'
    elif len(counts)==1: cfg['state']='REFERENCIA_DEGENERADA'
    if numeric and len(counts)>10:
        cfg['method']='quantiles'
        if counts.iloc[0]/len(finite)>=.85 and len(finite)-counts.iloc[0]>=MIN_EXPECTED_BIN_COUNT:
            cfg['dominant']=float(counts.index[0]); finite=finite[finite!=cfg['dominant']]
            cfg['method']='dominant_quantiles'
        # Quantile cuts retained only when both adjacent groups contain observations.
        cuts=np.unique(np.quantile(finite,np.linspace(0,1,n_bins+1)[1:-1]))
        cuts=cuts[(cuts>=finite.min()) & (cuts<finite.max())]
        retained=[]; previous=-np.inf
        for cut in cuts:
            if ((finite>previous)&(finite<=cut)).any(): retained.append(float(cut)); previous=cut
        cfg['edges']=[-np.inf]+retained+[np.inf]
    else:
        # Reference-only rare-category grouping, with a separately traced unseen bucket.
        keys=[token(v) for v in counts.index]
        cfg['categories']=keys[:50] if len(keys)>50 else keys
        cfg['rare_categories']=keys[50:]
    return cfg

def transform(s,cfg):
    s=pd.Series(s); result=pd.Series('__MISSING__',index=s.index,dtype=object)
    valid=s.notna()
    if not len(s): return result
    if cfg['numeric']:
        inf=valid & ~np.isfinite(s)
        result.loc[inf & (s>0)]='__POS_INF__'; result.loc[inf & (s<0)]='__NEG_INF__'
        valid &= ~inf
    if cfg['method']=='categories':
        def encode(v):
            k=token(v)
            return k if k in cfg['categories'] else ('__RARE__' if k in cfg.get('rare_categories',[]) else '__NEW__')
        result.loc[valid]=s.loc[valid].map(encode)
    else:
        dom=valid & (s==cfg['dominant']) if cfg['dominant'] is not None else pd.Series(False,index=s.index)
        result.loc[dom]='__DOMINANT__'; valid &= ~dom
        result.loc[valid]=pd.cut(s.loc[valid],cfg['edges'],labels=False).map(lambda x:'BIN:'+str(int(x)))
    return result

def calculate_psi(reference,current,binning_config,alpha=None,min_obs=None):
    alpha=ALPHA if alpha is None else alpha
    min_obs=MIN_MONTH_OBS if min_obs is None else min_obs
    if alpha<=0: raise ValueError('ALPHA debe ser positivo')
    r=transform(reference,binning_config); c=transform(current,binning_config)
    support=sorted(set(r)|set(c)|{'__MISSING__'})
    rc=r.value_counts().reindex(support,fill_value=0).to_numpy(float)
    cc=c.value_counts().reindex(support,fill_value=0).to_numpy(float)
    p=(rc+alpha)/(len(r)+alpha*len(support)); q=(cc+alpha)/(len(c)+alpha*len(support))
    terms=(q-p)*np.log(q/p)
    state='OK' if len(c)>=min_obs and len(r) else 'VOLUMEN_INSUFICIENTE'
    if binning_config['state']!='OK': state=binning_config['state']
    value=max(0,float(terms.sum())) if state=='OK' else np.nan
    return dict(PSI=value,PSI_missing_component=float(terms[support.index('__MISSING__')]) if state=='OK' else np.nan,
        Estado_calculo=state,N_bins_efectivos=int((rc>0).sum()),
        N_categorias_nuevas=int((c=='__NEW__').sum()),N_infinitos=int(c.isin(['__POS_INF__','__NEG_INF__']).sum()),
        Min_conteo_esperado=float((p*len(c))[rc>0].min()) if (rc>0).any() else np.nan,
        Categorias_raras_referencia=int(((rc>0)&(rc<MIN_EXPECTED_BIN_COUNT)).sum()),
        p=p,q=q,support=support,reference_counts=rc,current_counts=cc,contributions=terms)

def max_consecutive(values):
    vals=sorted(set(values)); best=run=0; prev=None
    for p in vals:
        run=run+1 if prev is not None and p.ordinal==prev.ordinal+1 else 1
        best=max(best,run); prev=p
    return best

def classify(n,total,reds,consecutive,degenerate=False):
    if degenerate or not n or n/total<MIN_TRAIN_MONTH_COVERAGE:return 'NO_EVALUABLE'
    if reds>=3 and reds/n>=.30:return 'CANDIDATA_DESCARTE'
    if consecutive>=2:return 'REVISION_PRIORITARIA'
    if reds:return 'EN_OBSERVACION'
    return 'APROBADA'

def consolidate(xtrain,variability,missing,summary):
    records=[]
    for col in xtrain:
        s=xtrain[col]; freq=s.value_counts()
        concentration=100*freq.iloc[0]/len(s) if len(freq) else np.nan
        records.append(dict(Variable=col,Filtro='Variabilidad',Metrica=concentration,Umbral=85,
            Clasificacion='Rechazado' if col in variability else 'Aprobado',Decision='Excluir' if col in variability else 'Conservar',Motivo='Filtro original TRAIN; denominador incluye nulos',Referencia='full_train'))
        if col not in variability:
            records.append(dict(Variable=col,Filtro='Nulos',Metrica=100*s.isna().mean(),Umbral=40,
                Clasificacion='Rechazado' if col in missing else 'Aprobado',Decision='Excluir' if col in missing else 'Conservar',Motivo='Filtro original TRAIN',Referencia='full_train'))
    for r in summary.to_dict('records'):
        records.append(dict(Variable=r['Variable'],Filtro='PSI mensual',Metrica=r['PSI_max_train'],Umbral=RED,Clasificacion=r['Estado_PSI'],Decision='Conservar para revision' if r['Estado_PSI']!='APROBADA' else 'Conservar',Motivo=r['Motivo_decision'],Referencia=r['Referencia_utilizada']))
    return pd.DataFrame(records).drop_duplicates(['Variable','Filtro'])

def run_analysis(df,matrices,reference_mode=REFERENCE_MODE,export=True):
    if not df.index.is_unique: raise ValueError('Indice original duplicado: alineacion ambigua')
    temporal=periods(df.periodo_observacion)
    if 'codmes' in df and not periods(df.codmes).equals(temporal): raise ValueError('codmes no coincide con periodo_observacion')
    for seg,x in matrices.items():
        if not x.index.is_unique or not x.index.isin(df.index).all(): raise ValueError('Indice invalido')
        if not df.loc[x.index,'SEGMENTO'].eq(seg).all(): raise ValueError('Segmento desalineado')
        pd.testing.assert_frame_equal(x,df.loc[x.index,x.columns])
    train=matrices['TRAIN']; months=sorted(temporal.loc[train.index].unique())
    if reference_mode not in ['early_train','full_train']: raise ValueError('Modo referencia invalido')
    ref_months=months[:REFERENCE_N_MONTHS] if reference_mode=='early_train' else months
    ref_idx=train.index[temporal.loc[train.index].isin(ref_months)]
    if len(ref_idx)<REFERENCE_MIN_N: raise ValueError('Referencia insuficiente: considerar full_train o ampliar meses previamente, sin optimizar PSI')
    rows=[]; refs=[]; summaries=[]; bins={}; detail=[]
    for col in train:
        ref=train.loc[ref_idx,col]; cfg=learn_bins(ref); bins[col]=cfg
        reftext=reference_mode+':'+','.join(map(str,ref_months))
        refs.append(dict(Variable=col,Referencia_utilizada=reftext,N_observaciones=len(ref),N_clientes=df.loc[ref_idx,'cliente_id'].nunique(),Pct_nulos=100*ref.isna().mean(),N_bins=len(set(transform(ref,cfg))),Estado=cfg['state'],Metodo_binning=cfg['method']))
        for seg,x in matrices.items():
            for month,idx in temporal.loc[x.index].groupby(temporal.loc[x.index]).groups.items():
                cur=x.loc[idx,col]; calc=calculate_psi(ref,cur,cfg)
                d={k:v for k,v in calc.items() if k not in ['p','q','support','reference_counts','current_counts','contributions']}
                d.update(Variable=col,Tipo='Numerica' if cfg['numeric'] else 'Categorica',Segmento=seg,Periodo=str(month),N_observaciones=len(cur),N_nulos=int(cur.isna().sum()),Pct_nulos=100*cur.isna().mean(),Pct_nulos_referencia=100*ref.isna().mean(),Pct_nulos_periodo=100*cur.isna().mean(),Diferencia_nulos_pp=100*(cur.isna().mean()-ref.isna().mean()),Metodo_binning=cfg['method'],Referencia_utilizada=reftext,Mes_en_referencia=seg=='TRAIN' and month in ref_months)
                d['Nivel_alerta']='NO_EVALUABLE' if not np.isfinite(d['PSI']) else ('ALERTA_ROJA' if d['PSI']>RED else 'ALERTA_AMBAR' if d['PSI']>=AMBER else 'ESTABLE')
                d['PSI_alpha_01']=calculate_psi(ref,cur,cfg,alpha=.1)['PSI']; d['PSI_alpha_1']=calculate_psi(ref,cur,cfg,alpha=1)['PSI']
                d['Diagnostico_calidad']='INFINITOS: categorias separadas; revisar origen' if d['N_infinitos'] else ''
                if d['Min_conteo_esperado']<MIN_EXPECTED_BIN_COUNT: d['Diagnostico_calidad']+='; soporte raro/conteo esperado <5, interpretar con sensibilidad alpha'
                rows.append(d)
                for j,label in enumerate(calc['support']):detail.append(dict(Variable=col,Segmento=seg,Periodo=str(month),Bin=label,N_referencia=calc['reference_counts'][j],N_actual=calc['current_counts'][j],P_referencia=calc['p'][j],P_actual=calc['q'][j],Contribucion=calc['contributions'][j],Estado_calculo=calc['Estado_calculo']))
    monthly=pd.DataFrame(rows).sort_values(['Variable','Segmento','Periodo'])
    for col,g in monthly.groupby('Variable',sort=False):
        t=g[g.Segmento=='TRAIN']; valid=t[t.Estado_calculo=='OK']; red=valid[valid.PSI>RED]; n=len(valid); consecutive=max_consecutive([pd.Period(p,'M') for p in red.Periodo]); state=classify(n,len(months),len(red),consecutive,bins[col]['state']!='OK')
        summaries.append(dict(Variable=col,Tipo=g.Tipo.iloc[0],PSI_max_train=valid.PSI.max(),PSI_promedio_train=valid.PSI.mean(),PSI_max_test=g.loc[g.Segmento=='TEST','PSI'].max(),PSI_max_oot=g.loc[g.Segmento=='OOT','PSI'].max(),Meses_train_evaluados=n,Meses_train_no_evaluables=len(months)-n,Meses_train_ambar=int(valid.PSI.between(AMBER,RED).sum()),Meses_train_rojos=len(red),Porcentaje_meses_rojos=100*len(red)/n if n else np.nan,Max_meses_rojos_consecutivos=consecutive,Periodos_criticos=','.join(red.Periodo),Cobertura_temporal=n/len(months),Estado_PSI=state,Motivo_decision=f'{n}/{len(months)} meses evaluables; {len(red)} rojos; racha {consecutive}; referencia {bins[col]["state"]}',Candidata_descarte=state=='CANDIDATA_DESCARTE',Referencia_utilizada=g.Referencia_utilizada.iloc[0]))
    summary=pd.DataFrame(summaries)
    references = pd.DataFrame(refs)
    # Keep audit artifacts in memory so later exports never recalculate PSI.
    monthly.attrs["psi_artifacts"] = {
        "bins": bins,
        "distributions": detail,
        "configuration": get_configuration(reference_mode),
    }
    if export:  # Backward-compatible API; interactive notebook uses export=False.
        export_psi_tables(monthly, summary, references)
    return monthly, summary, references


COLORS={'TRAIN':'#1769aa','TEST':'#ed8b23','OOT':'#338b55'}
def plot_psi_variable(variable,df_psi_mensual,summary=None,ax=None,show_legend=True):
    if ax is None:fig,ax=plt.subplots(figsize=(13,4))
    else:fig=ax.figure
    g=df_psi_mensual[df_psi_mensual.Variable==variable]
    calendar=pd.period_range(df_psi_mensual.Periodo.min(),df_psi_mensual.Periodo.max(),freq='M').astype(str)
    for seg,color in COLORS.items():
        s=g[g.Segmento==seg].set_index('Periodo').PSI.reindex(calendar)
        ax.plot(calendar,s,marker='.',label=seg,color=color)
        if seg=='TRAIN':
            for p,val in s[s>RED].items():ax.annotate(p,(p,val),fontsize=6,rotation=30)
    ax.axhline(AMBER,color='#c69200',ls='--');ax.axhline(RED,color='red',ls='--')
    state=summary.set_index('Variable').loc[variable,'Estado_PSI'] if summary is not None else ''
    import textwrap
    ax.set_title('\n'.join(textwrap.wrap(variable,48))+'\n'+state,fontsize=8);ax.set_ylim(bottom=0);ax.tick_params(axis='x',rotation=70,labelsize=6);ax.set_ylabel('PSI');ax.set_xlabel('periodo_observacion')
    if show_legend: ax.legend(fontsize=6)
    return fig,ax


PSI_STATES = ('APROBADA', 'EN_OBSERVACION', 'REVISION_PRIORITARIA',
              'CANDIDATA_DESCARTE', 'NO_EVALUABLE')


def plot_psi_dashboard(df_psi_mensual, df_resumen_psi, estado='TODAS',
                       columnas=4, por_pagina=12):
    """Build dashboard pages from memory; return figures without saving/showing."""
    if estado not in ('TODAS',) + PSI_STATES:
        raise ValueError(f'Estado desconocido: {estado}')
    if (isinstance(columnas, bool) or not isinstance(columnas, int) or columnas < 1
            or isinstance(por_pagina, bool) or not isinstance(por_pagina, int)
            or not 1 <= por_pagina <= 12):
        raise ValueError('Columnas positivas y por_pagina entre 1 y 12')
    selected = df_resumen_psi if estado == 'TODAS' else df_resumen_psi[
        df_resumen_psi.Estado_PSI == estado]
    variables = selected.Variable.tolist()
    figures = []
    for start in range(0, len(variables), por_pagina):
        page = variables[start:start + por_pagina]
        rows = math.ceil(len(page) / columnas)
        figure, axes = plt.subplots(rows, columnas,
                                    figsize=(6 * columnas, 4.8 * rows),
                                    squeeze=False)
        for axis, variable in zip(axes.flat, page):
            plot_psi_variable(variable, df_psi_mensual, df_resumen_psi,
                              ax=axis, show_legend=False)
        for axis in list(axes.flat)[len(page):]:
            axis.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        from matplotlib.lines import Line2D
        handles += [Line2D([], [], color='#c69200', ls='--'),
                    Line2D([], [], color='red', ls='--')]
        labels += [f'Umbral {AMBER:.2f}', f'Umbral {RED:.2f}']
        figure.legend(handles, labels, loc='upper center', ncol=5,
                      bbox_to_anchor=(.5, .975), frameon=False)
        figure.suptitle(f'PSI mensual — {estado} — página {len(figures) + 1}',
                       fontsize=13, y=.998)
        figure.tight_layout(rect=(0, 0, 1, .94), h_pad=2.2, w_pad=2)
        figures.append(figure)
    return figures


def plot_psi_heatmap(df_psi_mensual, segmento='TRAIN'):
    """Build one segment heatmap; missing periods remain masked, never zero."""
    if segmento not in COLORS:
        raise ValueError('Segmento desconocido')
    calendar = pd.period_range(df_psi_mensual.Periodo.min(),
                               df_psi_mensual.Periodo.max(), freq='M').astype(str)
    table = df_psi_mensual[df_psi_mensual.Segmento == segmento].pivot(
        index='Variable', columns='Periodo', values='PSI')
    table = table.reindex(index=sorted(df_psi_mensual.Variable.unique()),
                          columns=calendar)
    figure, axis = plt.subplots(figsize=(17, max(5, len(table) * .32)))
    cmap = ListedColormap(['#a3d9a5', '#ffcf66', '#e36a68'])
    cmap.set_bad('#d0d0d0')
    axis.imshow(np.ma.masked_invalid(table.to_numpy()), aspect='auto',
                cmap=cmap, norm=BoundaryNorm([0, AMBER, RED + 1e-12, np.inf], 3))
    axis.set_yticks(range(len(table)), table.index, fontsize=6)
    axis.set_xticks(range(len(table.columns)), table.columns, rotation=70)
    axis.set_title(f'PSI {segmento}: verde <{AMBER:.2f}; ámbar <={RED:.2f}; '
                   f'rojo >{RED:.2f}; gris sin evaluación')
    figure.tight_layout()
    return figure, axis


def _backup_exports(root, folders):
    """Preserve existing outputs before an explicit export."""
    import shutil
    from datetime import datetime
    existing = [root / name for name in folders if (root / name).exists()]
    if existing:
        archive = root / 'evidencias' / ('respaldo_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        for folder in existing:
            shutil.copytree(folder, archive / folder.name)


def export_psi_tables(df_psi_mensual, df_resumen_psi, df_referencias_psi,
                      df_reporte_univariado=None, *, root=None, backup=True):
    """Write calculated tables and cached audit artifacts, without recalculation."""
    root = ROOT if root is None else Path(root)
    if backup:
        _backup_exports(root, ['reportes'])
    folder = root / 'reportes'
    folder.mkdir(parents=True, exist_ok=True)
    tables = {'df_psi_mensual': df_psi_mensual,
              'df_resumen_psi': df_resumen_psi,
              'df_reporte_psi': df_resumen_psi,
              'referencias': df_referencias_psi}
    if df_reporte_univariado is not None:
        tables['df_reporte_univariado'] = df_reporte_univariado
    artifacts = df_psi_mensual.attrs.get('psi_artifacts', {})
    if 'distributions' in artifacts:
        tables['distribuciones_bins'] = pd.DataFrame(artifacts['distributions'])
    paths = []
    for name, table in tables.items():
        path = folder / f'{name}.csv'
        table.to_csv(path, index=False)
        paths.append(path)
    for key, name in [('bins', 'bins.json'), ('configuration', 'parametros.json')]:
        if key in artifacts:
            path = folder / name
            path.write_text(json.dumps(artifacts[key], indent=2, allow_nan=True),
                            encoding='utf-8')
            paths.append(path)
    return paths


def export_psi_figures(df_psi_mensual, df_resumen_psi, *, individuales=False,
                       dashboards=True, heatmaps=True, estados=('TODAS',),
                       columnas=4, por_pagina=12, root=None, backup=True):
    """Save requested figures only; all plotting uses current in-memory tables."""
    root = ROOT if root is None else Path(root)
    folders = (['graficos_individuales'] if individuales else [])
    folders += ['subplots'] if dashboards else []
    folders += ['heatmaps'] if heatmaps else []
    if backup:
        _backup_exports(root, folders)
    paths, manifest = [], []
    for folder in folders:
        (root / folder).mkdir(parents=True, exist_ok=True)
    if individuales:
        for index, variable in enumerate(df_resumen_psi.Variable, 1):
            figure, _ = plot_psi_variable(variable, df_psi_mensual, df_resumen_psi)
            figure.tight_layout()
            name = f'{index:03d}_{variable[:100]}.png'
            path = root / 'graficos_individuales' / name
            figure.savefig(path, dpi=160)
            plt.close(figure)
            paths.append(path)
            manifest.append({'Variable': variable, 'Archivo': name})
        (root / 'reportes').mkdir(parents=True, exist_ok=True)
        pd.DataFrame(manifest).to_csv(root / 'reportes/graficos.csv', index=False)
    if dashboards:
        for state in estados:
            figures = plot_psi_dashboard(df_psi_mensual, df_resumen_psi,
                                          state, columnas, por_pagina)
            for page, figure in enumerate(figures, 1):
                path = root / 'subplots' / f'{state}_{page}.png'
                figure.savefig(path, dpi=140)
                plt.close(figure)
                paths.append(path)
    if heatmaps:
        for segment in COLORS:
            figure, _ = plot_psi_heatmap(df_psi_mensual, segment)
            path = root / 'heatmaps' / f'{segment}.png'
            figure.savefig(path, dpi=160)
            plt.close(figure)
            paths.append(path)
    return paths


def export_plots(monthly, summary):
    """Compatibility wrapper: export all figure types as in the previous API."""
    return export_psi_figures(monthly, summary, individuales=True,
                              estados=('TODAS',) + PSI_STATES)

