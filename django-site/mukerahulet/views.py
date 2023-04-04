from django.shortcuts import render
from utils.util import calcualte_maze_points
def home(request):
    sap, sar, aar, percentage=calcualte_maze_points()
    #sap = [['C', 'C'], ['C', 'C']]
    #sar = [['C', 'C'], ['C', 'C'], ['C', 'C'], ['C', 'C'], ['A', 'A'], ['C', 'C']]
    #aar = [['A', 'C', 'C'], ['C', 'C', 'A'], ['C', 'A', 'A'], ['A', 'A', 'C'], ['A', 'C', 'A'], ['B', 'C', 'C'], ['C', 'C', 'A'], ['C', 'A', 'C']]

    return render(request, 'bkp.html',{'sap':sap, 'sar':sar, 'aar':aar})