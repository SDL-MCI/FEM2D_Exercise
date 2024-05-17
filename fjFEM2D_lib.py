import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap

import re
from typing import Tuple


def read_ANSYS(file_path: str | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read an input file provided by ANSYS.

    Extracts the nodal coordinates, element connectivity, boundary conditions, material properties, and external forces from the selected file:

    N, E, BC, matProp, elMat, fex

    Parameters
    ----------
    file_path : str | None
        Path of the file to read. Opens a file dialog if None.

    Returns
    ----------
    N : np.ndarray
        Array of node coordinates.
    E : np.ndarray
        2D array of element nodes. Each row corresponds to an element.
    BC : np.ndarray
        Array of boundary conditions.
    matProp : np.ndarray
        Array of material properties used in the analysis.
    elMat : np.ndarray
        Array of material indices for each element.
    fex : np.ndarray
        Array of external forces.
    """

    if file_path is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.wm_attributes('-topmost', 1)
        root.withdraw()
        file_path = tk.filedialog.askopenfilename(title="Select File", filetypes=[("All Files", "*.dat*")])

    # Open the file for reading
    with open(file_path, 'r') as file:
        # Read all lines from the file
        data = file.readlines()

    # number of lines
    numL = len(data)

    # search keywords
    lineNumNode = 0
    lineNumElement  = []
    lineNset = []
    lineD = []
    lineMatW  = []
    lineMatM = []
    lineF = []

    for i in range(0,numL):

        currL = data[i]

        if currL[:6] == "nblock":
            lineNumNode = i

        if currL[:6] == "eblock":
            lineNumElement.append(i)

        if currL[:7] == "CMBLOCK":
            lineNset.append(i)

        if currL[:2] == "d,":
            lineD.append(i)

        if currL[:7] == "MP,KXX,":
            lineMatW.append(i)

        if currL[:6] == "MP,EX,":
            lineMatM.append(i)

        if currL[:4] == "sfe,":
            lineF.append(i)

    # read nodes
    e = 2
    coords = []
    while True:
        currL = data[lineNumNode+e]
        if  currL[:2] == "-1" :
            break

        coords.append([float(currL[13:29]),float(currL[31:49])])

        e = e + 1

    N = np.array(coords)

    # read elements
    elements = []
    elMat = []
    for i in range(0,len(lineNumElement)):
        # analyse type
        if i == 0:
            currL = data[lineNumElement[i]-2]
            ind = [index for index, char in enumerate(currL) if char == ","]
            etypeNum = int(currL[ind[1]+1:-1])

            if etypeNum == 182 or etypeNum == 183:
                # mechanical analysis
                antype = 0
            else:
                # temp or electro
                antype = 1

        e = 2
        while True:
            currL = data[lineNumElement[i]+e]
            if  currL[:2] == "-1" :
                break

            if len(currL) >= 80:
                elNum = currL[100:]
                # Extract all integers from the string using regular expression
                numbers_list_of_strings = re.findall(r'\d+', elNum)

                # Convert the list of strings to a list of integers
                elNumbers = [int(num) for num in numbers_list_of_strings]

                elements.append(elNumbers)
                elMat.append(int(currL[:9]))

            e = e + 1

    elMat = np.array(elMat) #list to np.array

    # delete double columns
    numNE = len(elements[1])
    
    if numNE == 8:
        indD = [3, 6]
    else:
        indD = [3]

    E = np.array(elements, dtype="int")
    E = np.delete(E, indD, axis=1)

    # read material parameters
    if antype == 0:
        # mechanical
        matProp = np.zeros((len(lineMatM),2))

        for i in range(0,len(lineMatM)):
            currL = data[lineMatM[i]]
            ind = [index for index, char in enumerate(currL) if char == ","]
            matProp[i,0] = float(currL[ind[2]+1:ind[3]])

            currL = data[lineMatM[i]+1]
            ind = [index for index, char in enumerate(currL) if char == ","]
            matProp[i,1] = float(currL[ind[2]+1:ind[3]])
        

    else:
        # thermal or eletric
        matProp = np.zeros((len(lineMatM),2))

        for i in range(0,len(lineMatW)):

            currL = data[lineMatW[i]]
            ind = [index for index, char in enumerate(currL) if char == ","]
            
            matProp[i,0] = float(currL[ind[2]+1:ind[3]])

        matProp = np.array(matProp)

    # find nset
    NsetName = []
    NsetNodes = []

    numNSet = len(lineNset)

    for i in range(0,numNSet):
        currL = data[lineNset[i]]
        name = currL[8:15]

        nodes = []
        for k in range(0,100):

            currL2 = data[lineNset[i]+2+k]
            if currL2[0]=="/":
                break

            # Extract all integers from the string using regular expression
            numbers_list_of_strings = re.findall(r'\d+', currL2)

            # Convert the list of strings to a list of integers
            nodes.append([int(num) for num in numbers_list_of_strings])

        NsetName.append(name.replace(" ",""))
        nodes = [item for sublist in nodes for item in sublist]
        NsetNodes.append(nodes)

    # boundary conditions
    numBC = len(lineD)
    BCnodes = []
    BCmag = []

    if antype == 0:
        #mechanical
        nodes = []
        constrDOF = []

        
        for i in range(0,len(lineD)):
            currL = data[lineD[i]]
            if currL[6:8] == "ux":
                cDOF = 0
            if currL[6:8] == "uy":
                cDOF = 1

            e = 2
            while True:
                currL = data[lineD[i] - e]

                if currL[:1] == "(":
                    break

                # Extract all integers from the string using regular expression
                numbers_list_of_strings = re.findall(r'\d+', currL)

                # Convert the list of strings to a list of integers
                nNumbers = [int(num) for num in numbers_list_of_strings]
                
                nodes.append(nNumbers)
                constrDOF.append([cDOF]*len(nNumbers))

                e = e + 1

        nodes = [item for sublist in nodes for item in sublist]
        constrDOF = [item for sublist in constrDOF for item in sublist]
        BC = np.zeros((len(nodes),2))
        BC[:,0] = np.array(nodes, dtype="int")
        BC[:,1] = np.array(constrDOF, dtype="int")
        
    else:
        # thermal or electric
        
        for i in range(0,numBC):

            currL = data[lineD[i]]
            name = currL[2:7]
            
            # search nset
            for j in range(0,numNSet):
                if name == NsetName[j]:
                    nodes = NsetNodes[j]

            ind = [k for k, char in enumerate(currL) if char == ","]
            mag = float(currL[ind[-1]+1:])

            BCnodes.append(nodes)
            BCmag.append([mag]*len(nodes))

        BC = []
        BC.append(BCnodes)
        BC.append(BCmag)
        BC = [item for sublist in BC for item in sublist]
        BC = [item for sublist in BC for item in sublist]

        n = int(len(BC)/2)
        BC = np.reshape(BC, (n, 2), order='F')

    fex = np.array([0]) #was 0 before
    # external force
    if antype == 0:
        
        numEL = E.shape[0]
        # find nodes
        nodesF = []
        for i in range(0,len(lineNumElement)):

            ind = lineNumElement[i]+2
            currL = data[ind]

            e = 0
            if int(currL[0:9])>numEL:

                while True:
                    currL = data[ind+e] 

                    if currL[:2] == "-1":
                        break

                    # Extract all integers from the string using regular expression
                    numbers_list_of_strings = re.findall(r'\d+', currL)
                    # Convert the list of strings to a list of integers
                    nodes = [int(num) for num in numbers_list_of_strings]

                    nodesF.append(nodes[5:])

                    e = e + 1

        # magnitude of force in global x and y direction
        magF = []
        for i in range(0,len(lineF)):
            currL = data[lineF[i]]
            ind = [k for k, char in enumerate(currL) if char == ","]
            magF.append((float(currL[ind[4]+1:-1])))

        # determine external force vector
        fex = np.zeros((N.shape[0]*2))
      
        for i in range(0,len(nodesF)):

            # nodes
            ind = np.array(nodesF[i],dtype="int")
            # element length
            dx = np.abs(N[ind[0]-1,0] - N[ind[1]-1,0])
            dy = np.abs(N[ind[0]-1,1] - N[ind[1]-1,1])
            le = np.sqrt(dx**2 + dy**2)
            
            px = magF[0]
            py = magF[1]

            # assemble force vector
            if len(ind)==2:
                # linear element
                dof = np.array([ 2*ind[0]-2,  2*ind[0]-1, 2*ind[1]-2, 2*ind[1]-1 ])
                fex[dof] += 0.5*le*np.array([px, py, px, py])
            else:
                # quadratic element
                X = N[ind-1,:]

                # Simpson integration
                xiIP = [0.0, 1/3, 2/3, 1.0]
                w    = [1/8, 3/8, 3/8, 1/8]

                fexe = np.zeros(6)

                # numerical integragtion
                for i in range(0,len(w)):

                    # integration point
                    xi = xiIP[i]

                    # shape functions
                    N1 = -xi + 2*xi**2
                    N2 = 1 - 3*xi + 2*xi**2
                    N3 = 4*xi - 4*xi**2

                    # local derivatives
                    dN1dxi = -1 + 4*xi
                    dN2dxi = -3 + 4*xi
                    dN3dxi = 4 - 8*xi

                    # determinant of jacobian
                    dxdxi = dN1dxi*X[0,0] + dN2dxi*X[1,0] + dN3dxi*X[2,0]
                    dydxi = dN1dxi*X[0,1] + dN2dxi*X[1,1] + dN3dxi*X[2,1]
                    detJ  = np.sqrt(dxdxi**2 + dydxi**2)

                    # integration
                    fexe += np.array([px*N1,py*N1,px*N2,py*N2,px*N3,py*N3])*w[i]*detJ


                dof = np.array([ 2*ind[0]-2,  2*ind[0]-1, 2*ind[1]-2, 2*ind[1]-1, 2*ind[2]-2, 2*ind[2]-1 ])
                fex[dof] += fexe    
                

    return N, E, BC, matProp, elMat, fex

def plot_mesh(N, E, BC, elMat):
    """Plot the discretization of the mesh.

    Plot the provided mesh and the nodes on which boundary conditions are applied.:

    Parameters
    ----------
    N : np.ndarray
        Array of node coordinates.
    E : np.ndarray
        2D array of element nodes. Each row corresponds to an element.
    BC : np.ndarray
        Array of boundary conditions.
    matProp : np.ndarray
        Array of material properties used in the analysis.
    elMat : np.ndarray
        ??
    """

    colors = [(0., 0.6, 0.6),(0.6, 0.6, 0.),(0., 0., 0.6)]

    numNE = E.shape[1]
    # LST
    if numNE == 6:
        indE = [0, 3, 1, 4, 2, 5, 0]
    # CST
    else:
        indE = [0, 1, 2]

    for i in range(0,E.shape[0]):
        ind = np.resize(E[i,:]-1,numNE)

        x = np.resize(N[ind[indE],0],numNE+1)
        y = np.resize(N[ind[indE],1],numNE+1)

        plt.fill(x, y, color=colors[elMat[i]-1], linewidth = 1, edgecolor = "k", alpha = 0.5)

    # plot nodes
    numN = N.shape[0]
    x = np.resize(N[:,0],numN)
    y = np.resize(N[:,1],numN)
    plt.scatter(x, y, color='k', marker='o', label='Points',s=4)

    # plot boundary conditons
    ind = [int(value) for value in BC[:,0]-1]
    x = np.resize(N[ind,0],len(ind))
    y = np.resize(N[ind,1],len(ind))
    plt.scatter(x, y, color='red', marker='o', label='Points',s=13)

    # axis properties
    plt.axis("equal")
    plt.axis("off")
    plt.show()

def post_proc(N: np.ndarray, E: np.ndarray, u: np.ndarray):
    """Postprocessing for the 2D FEM analysis.

    Parameters
    ----------
    N : np.ndarray
        Array of node coordinates.
    E : np.ndarray
        2D array of element nodes. Each row corresponds to an element.
    u : np.ndarray
        Array of primary unknowns returned from the FEM solver.
    """

    # post-processing
    numN = N.shape[0]

    x = np.resize(N[:,0],numN)
    y = np.resize(N[:,1],numN)

    # min and max value of results
    minU = np.min(u)
    maxU = np.max(u)
    myLevels = np.linspace(minU, maxU, num=10)
    # ANSYS colors
    myColors = np.array([[0, 0, 255],
                         [0, 178, 255],
                         [0, 255, 255],
                         [0, 255, 178],
                         [0, 255, 0],
                         [178, 255, 0],
                         [255, 255, 0],
                         [255, 178, 0],
                         [255, 0, 0],], dtype=float)/255

    myCmap = ListedColormap(myColors)

    # linear elment
    if E.shape[1] == 3:
        E2 = E
        triang = mtri.Triangulation(x,y,E-1)

        contour = plt.tricontourf(triang, u, vmin = minU, vmax = maxU, levels=myLevels, cmap=myCmap)
        plt.triplot(triang, color="k", linewidth=0.5)
 
    # quadratic element
    else:
        n = 4*E.shape[0]
        E2 = np.zeros((4*E.shape[0],3))
        e = 0
        for i in range(0,E.shape[0]):
            ind = np.resize(E[i,:]-1,6)
            E2[e,:]   = [ind[0], ind[3], ind[5]]
            E2[e+1,:] = [ind[3], ind[1], ind[4]]
            E2[e+2,:] = [ind[3], ind[4], ind[2]]
            E2[e+3,:] = [ind[3], ind[2], ind[5]]
            e = e + 4

        triang = mtri.Triangulation(x,y,E2)
        contour = plt.tricontourf(triang, u, vmin = minU, vmax = maxU, levels=myLevels, cmap=myCmap)

        # plot mesh
        indE = [0, 3, 1, 4, 2, 5, 0]

        for i in range(0,E.shape[0]):
            ind = np.resize(E[i,:]-1,6)

            xe = np.resize(N[ind[indE],0],7)
            ye = np.resize(N[ind[indE],1],7)

            plt.plot(xe,ye,color="k",linewidth = 0.5)

    # plot nodes
    plt.scatter(x, y, color='k', marker='o', label='Points',s=2)
    plt.axis("equal")
    plt.axis("off")
    plt.colorbar(contour)

    plt.show()

if __name__ == "__main__":
    pass
    # read ANSYS input file
    #N, E, BC, matProp, elMat, fex = read_ANSYS()

    # plot discretization
    #plot_mesh(N, E, BC, elMat)

    # FEM
    #u = solve_fem_2D(N, E, BC, matProp, elMat)

    # plot result
    #post_proc(N, E, u)