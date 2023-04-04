from django.shortcuts import render
from utils.util import calcualte_maze_points
def home(request):
    sap, sar, aar, percentage=calcualte_maze_points()
    return render(request, 'bkp.html',{'sap':sap, 'sar':sar, 'aar':aar})
