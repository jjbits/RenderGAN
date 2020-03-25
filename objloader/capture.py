# sketchnet model loder and capturing tool 

import plotly
import numpy as np
import meshio
import plotly.graph_objects as go
from numpy import sin, cos, pi
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot
from os import walk
import shutil
import pickle
from datetime import datetime
import time

plotly.__version__
init_notebook_mode(connected=True)


# generate rotation matrices
def rot_x(t):
    return np.array([[1, 0, 0],
                     [0, cos(t), -sin(t)],
                     [0, sin(t), cos(t)]])


def rot_z(t):
    return np.array([[cos(t), -sin(t), 0],
                     [sin(t), cos(t), 0],
                     [0, 0, 1]])


def rot_y(t):
    return np.array([[cos(t), 0, sin(t)],
                     [0, 1, 0],
                     [-sin(t), 0, cos(t)]])


# given 'obj' data, conver to plotly mesh3d data
def obj_data_to_mesh3d(odata):
    # odata is the string read from an obj file
    vertices = []
    faces = []
    lines = odata.splitlines()

    for line in lines:
        slist = line.split()
        if slist:
            if slist[0] == 'v':
                vertex = np.array(slist[1:], dtype=float)
                vertices.append(vertex)
            elif slist[0] == 'f':
                face = []
                for k in range(1, len(slist)):
                    face.append([int(s) for s in slist[k].replace('//', '/').split('/')])
                if len(face) > 3:  # triangulate the n-polyonal face, n>3
                    faces.extend(
                        [[face[0][0] - 1, face[k][0] - 1, face[k + 1][0] - 1] for k in range(1, len(face) - 1)])
                else:
                    faces.append([face[j][0] - 1 for j in range(len(face))])
            else:
                pass

    return np.array(vertices), np.array(faces)


def get_layout(camera):
    layout = go.Layout(
        font=dict(size=16, color='white'),
        width=512,
        height=512,
        scene=dict(xaxis=dict(visible=False),
                   yaxis=dict(visible=False),
                   zaxis=dict(visible=False),
                   # aspectratio=dict(x=1.5*1,
                   #                 y=0.25*1,
                   #                 z=2.0*1),
                   aspectratio=dict(x=2.0,
                                    y=0.65,
                                    z=2.0),
                   camera=camera,
                   ),
        paper_bgcolor='rgb(255,255,255)',
        margin=dict(l=10,
                    r=10,
                    b=10,
                    t=10,
                    pad=4),
    )
    return layout


# render pipeline
def render_pipeline(vertices, triangles):
    # apply rot_z(angle2) * rot_x(angle1)
    # A = rot_x(pi/4)
    # print(A)
    # B = rot_z(pi/4)
    # print(B)

    # Apply the product of the two rotations to the object vertices:
    # BA = np.dot(B,A)
    # print(BA)
    # new_vertices = np.einsum('ik, kj -> ij',  vertices, BA.T) # new_vertices has the shape (n_vertices, 3)

    # x, y, z = new_vertices.T
    # tri_points = new_vertices[triangles]
    x, y, z = vertices.T
    tri_points = vertices[triangles]

    I, J, K = triangles.T

    pl_mygrey = [0, 'rgb(153, 153, 153)'], [1., 'rgb(255,255,255)']

    pl_mesh = go.Mesh3d(x=x,
                        y=y,
                        z=z,
                        colorscale=pl_mygrey,
                        intensity=z,
                        flatshading=True,
                        i=I,
                        j=J,
                        k=K,
                        name='RenderGAN',
                        showscale=False)

    pl_mesh.update(cmin=-7,  # atrick to get a nice plot (z.min()=-3.31909)
                   lighting=dict(ambient=0.18,
                                 diffuse=1,
                                 fresnel=0.1,
                                 specular=1,
                                 roughness=0.05,
                                 facenormalsepsilon=1e-15,
                                 vertexnormalsepsilon=1e-15),
                   lightposition=dict(x=100,
                                      y=200,
                                      z=0)
                   );

    Xe = []
    Ye = []
    Ze = []
    for T in tri_points:
        Xe.extend([T[k % 3][0] for k in range(4)] + [None])
        Ye.extend([T[k % 3][1] for k in range(4)] + [None])
        Ze.extend([T[k % 3][2] for k in range(4)] + [None])

    # define the trace for triangle sides
    lines = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        name='',
        line=dict(color='rgb(70,70,70)', width=1))

    return pl_mesh, lines


def caputre_image(pl_mesh, lines, file_id, save_path):
    # Initial position y: +1 z: -1 x: 0 Seeing the front of the plane
    angles = {}
    # 0 -> nose left, view plane side: up=dict(x=0, y=1, z=0) eye=dict(x=-2.5, y=0, z=0.0)
    angles[0] = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=-2.5, y=0.0, z=0.0))
    # 1 -> nose right, view plane side: up=dict(x=0, y=1, z=0) eye=dict(x=2.5, y=0, z=0.0)
    angles[1] = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=2.5, y=0.0, z=0.0))
    # 2 -> nose left, view plane bottom: up=dict(x=-1, y=0, z=-1) eye=dict(x=0, y=-2.5, z=0.0)
    angles[2] = dict(up=dict(x=-1, y=-1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=-2.5, z=0.0))
    # 3 -> from top plane nose down: up=dict(x=0, y=0, z=1) eye=dict(x=0, y=2.5, z=0)
    angles[3] = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=2.5, z=0.0))
    # 4 -> nose front, view plane side: up=dict(x=0, y=1, z=0) eye=dict(x=0.0, y=0, z=-2.5)
    angles[4] = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0.0, y=0.0, z=-2.5))
    # 5 -> nose back, view plane side: up=dict(x=0, y=1, z=0) eye=dict(x=0.0, y=0, z=2.5)
    angles[5] = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0.0, y=0.0, z=2.5))
    # 6 -> from initial position, 30 degree up to y: up=dict(x=0, y=1, z=0), eye=dict(x=0, y=1.25, z=-2.17)
    angles[6] = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0.0, y=1.25, z=-2.17))
    # 7 -> from 6, z to x cos(55)=1.25, cos(35)=1.78 degrees: up=dict(x=0, y=1, z=0), eye=dict(x=1.78, y=1.25, z=-1.25)
    angles[7] = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=1.78, y=1.25, z=-1.25))
    # 8 -> from 6, up=dict(x=0, y=1, z=0), eye=dict(x=2.17, y=1.25, z=0)
    angles[8] = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=2.17, y=1.25, z=0))
    # 9: up=dict(x=0, y=1, z=0), eye=dict(x=1.78, y=1.25, z=1.25)
    angles[9] = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=1.78, y=1.25, z=1.25))
    # 10 -> from 6, 30 degree up to y, z = -z: up=dict(x=0, y=1, z=0), eye=dict(x=0, y=1.25, z=2.17)
    angles[10] = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=1.25, z=2.17))
    # 11 -> up=dict(x=0, y=1, z=0), eye=dict(x=-1.78, y=1.25, z=1.25)
    angles[11] = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=-1.78, y=1.25, z=1.25))
    # 12 -> up=dict(x=0, y=1, z=0), eye=dict(x=-2.17, y=1.25, z=0)
    angles[12] = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=-2.17, y=1.25, z=0))
    # 13 -> up=dict(x=0, y=1, z=0), eye=dict(x=-1.78, y=1.25, z=-1.25)
    angles[13] = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=-1.78, y=1.25, z=-1.25))

    image_name_pre = save_path + file_id
    image_name_pro = ".png"

    for i in range(14):
        print("capturing " + str(i))
        camera = angles[i]
        layout = get_layout(camera)
        fig = go.Figure(data=[pl_mesh, lines], layout=layout)
        name = image_name_pre + '-' + str(i) + image_name_pro
        fig.write_image(name, width=512, height=512, scale=1.0)


def show_obj(pl_mesh, lines):
    layout = go.Layout(font=dict(size=16, color='white'),
                       width=512,
                       height=512,
                       scene=dict(xaxis=dict(visible=False),
                                  yaxis=dict(visible=False),
                                  zaxis=dict(visible=False),
                                  # aspectratio=dict(x=1.5,
                                  #                 y=0.25,
                                  #                 z=2.0),
                                  aspectratio=dict(x=2.0,
                                                   y=0.65,
                                                   z=2.0),
                                  camera=dict(up=dict(x=0, y=1, z=0),
                                              center=dict(x=0, y=0, z=0),
                                              eye=dict(x=0, y=-2.5, z=0.0))
                                  ),
                       paper_bgcolor='rgb(255,255,255)',
                       )

    fig = go.Figure(data=[pl_mesh, lines], layout=layout)
    iplot(fig)


# iterate through the data folder and get the gold list to render
def get_obj_folder_list(mypath):
    searching_folder = 'screenshots'

    # traverse the folder and look for screencapture
    # mypath = './data/02691156/'
    f = []
    d = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        d.extend(dirnames)

    gold = []
    for each_dir in d:
        for (dirpath, dirnames, filenames) in walk(mypath + each_dir):
            if searching_folder in dirnames:
                gold.append(each_dir)
                # print(each_dir)

    return gold


def run_capture(obj_path, file_id, save_path):
    obj_data = open(obj_path).read()
    print("Got the obj file")
    vertices, triangles = obj_data_to_mesh3d(obj_data)
    print("loaded the obj")
    #time.sleep(3)
    pl_mesh, lines = render_pipeline(vertices, triangles)
    print("loaded the pipeline")
    #time.sleep(3)
    caputre_image(pl_mesh, lines, file_id, save_path)

def run_figure(obj_path):
    obj_data = open(obj_path).read()
    vertices, triangles = obj_data_to_mesh3d(obj_data)
    pl_mesh, lines = render_pipeline(vertices, triangles)
    show_obj(pl_mesh, lines)

data_path = '/Volumes/Data/projects/cs230/Project/RenderGAN/objloader/data/02691156/'
post_path = '/models/model_normalized.obj'
#save_path = '/Volumes/Data/projects/cs230/Project/RenderGAN/objloader/data/temp/'
save_path = '/Volumes/Data/projects/cs230/Project/RenderGAN/objloader/data/planes/train/A/'

#gold_list = get_obj_folder_list(data_path)
#gold_list = ['84a167f743ec4f4de6f558e7f4a68d3']

with open("./remain_list.txt", "rb") as fp:   #Pickling
    gold_list = pickle.load(fp)

total_model = len(gold_list)
print("Total Models To Render: " + str(total_model))

counter = 0
for each_gold in gold_list:
    print("Processing Model: " + each_gold)
    obj_name = data_path + each_gold + post_path
    run_capture(obj_name, each_gold, save_path)
    counter += 1
    remaining = total_model - counter
    print("Remaining: " + str(remaining) + "==> " + str(datetime.now()))
