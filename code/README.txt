
You can set the latent value (k) of NLLD and LLD models at line 21 in nlds.py.

You can execute by:
	./run.sh

There are 10 different runtypes:
	1: fitting with NLLD/LLD (our segmentation scheme)
	2: forecasting with NLLD/LLD (our segmentation scheme)
	3: fitting with NLLD/LLD (incremental segmentation)
	4: fitting with NLLD/LLD (single segmentation)
	5: forecasting with NLLD/LLD (single segmentaion)
	6: fitting with SIR (our segmentation scheme)
	7: forecasting with SIR (our segmentation scheme)
	8: fitting with SIR (single segmentation)
	9: forecasting with SIR (single segmentation)
	10: fitting with SIR (incremental segmentation)

You can specify the runtype in run.sh.
