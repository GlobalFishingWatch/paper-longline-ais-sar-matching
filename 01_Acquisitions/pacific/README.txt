### KSAT detection processing

1. convert detections .kmz --> csv
2. use process-ksat-detections-20190110.ipynb to:
		convert .kmz --> .kml
		extract .kml footprints as WKTs
		join, upload to BQ
