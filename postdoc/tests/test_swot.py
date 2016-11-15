import matplotlib.pyplot as plt
from postdoc import swot

def test_add_swot_swath():
	kml_file = "/data/SWOT/SWOT_CalVal_june2015_Swath_10_60.kml"
	plt.figure()
	swot.add_swot_swath(kml_file)
