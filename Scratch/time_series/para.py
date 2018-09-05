#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XDMF Reader'
skin_surfacexdmf = XDMFReader(FileNames=['/home/jakobes/opt/freesurfer/subjects/ZHI/skin_surface.xdmf'])

# Properties modified on skin_surfacexdmf
skin_surfacexdmf.GridStatus = ['Grid']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
renderView1.ViewSize = [2397, 1321]

# show data in view
skin_surfacexdmfDisplay = Show(skin_surfacexdmf, renderView1)
# trace defaults for the display properties.
skin_surfacexdmfDisplay.Representation = 'Surface'
skin_surfacexdmfDisplay.ColorArrayName = [None, '']
skin_surfacexdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
skin_surfacexdmfDisplay.SelectOrientationVectors = 'None'
skin_surfacexdmfDisplay.ScaleFactor = 0.0
skin_surfacexdmfDisplay.SelectScaleArray = 'None'
skin_surfacexdmfDisplay.GlyphType = 'Arrow'
skin_surfacexdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
skin_surfacexdmfDisplay.ScalarOpacityUnitDistance = 11.210020948853789
skin_surfacexdmfDisplay.SetScaleArray = [None, '']
skin_surfacexdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
skin_surfacexdmfDisplay.OpacityArray = [None, '']
skin_surfacexdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'

# reset view to fit data
renderView1.ResetCamera()


##############################
# Make spheres
##############################

with open("channel.pos", "r") as ifh:
    positions = [tuple(map(float, line.split()[1:])) for line in ifh.readlines()]

for xyz in positions:
    sphere1 = Sphere()
    # sphere1.Center = [x, y, z]
    sphere1.Center = list(map(lambda x: 10*x, xyz))
    sphere1.Radius = 4.0

    # show data in view
    sphere1Display = Show(sphere1, renderView1)
    # trace defaults for the display properties.
    sphere1Display.Representation = 'Surface'
    sphere1Display.ColorArrayName = [None, '']
    sphere1Display.OSPRayScaleArray = 'Normals'
    sphere1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    sphere1Display.SelectOrientationVectors = 'None'
    sphere1Display.ScaleFactor = 0.0
    sphere1Display.SelectScaleArray = 'None'
    sphere1Display.GlyphType = 'Arrow'
    sphere1Display.PolarAxes = 'PolarAxesRepresentation'
    sphere1Display.SetScaleArray = [None, '']
    sphere1Display.ScaleTransferFunction = 'PiecewiseFunction'
    sphere1Display.OpacityArray = [None, '']
    sphere1Display.OpacityTransferFunction = 'PiecewiseFunction'

    # change solid color
    sphere1Display.DiffuseColor = [0.6666666666666666, 0.0, 0.0]

# current camera placement for renderView1
renderView1.CameraPosition = [-172.67435420355594, 506.6777951965837, 256.36638608209915]
renderView1.CameraFocalPoint = [-1.2265659999999983, -0.2927814999999966, 2.11121]
renderView1.CameraViewUp = [0.09177096705004614, -0.42143621690140354, 0.9022026405916426]
renderView1.CameraParallelScale = 153.35092947425485

SaveScreenshot("foo.png")

