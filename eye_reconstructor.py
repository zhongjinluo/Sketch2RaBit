import openmesh as om
import numpy as np
import random
import pdb
from scipy.spatial.transform import Rotation as R
import time

class Surface():
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None
    
    def fit_surface(self,points):
        mean = np.mean(points,axis=0)
        points = points - mean
        u,s,v = np.linalg.svd(points)
        a,b,c = v[-1]
        d = -np.sum(v[-1]*mean)
        # ax+by+cz+d=0
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        if abs(self.d)<1e-4:
            print("small d!!!")
        else:
            self.a /= self.d
            self.b /= self.d
            self.c /= self.d
            self.d = 1
        return 
    
    def reflect_one_point(self,point):
        x,y,z = point
        t = (self.a*x+self.b*y+self.c*z+self.d)/(self.a**2+self.b**2+self.c**2)
        x -= self.a*t
        y -= self.b*t
        z -= self.c*t
        new_point = np.array([x,y,z])
        dist = t * np.sqrt(self.a**2+self.b**2+self.c**2)
        return new_point,t

    def reflect(self,points):
        nlist = []
        tlist = []
        for point in points:
            new_point, t = self.reflect_one_point(point)
            nlist.append(new_point)
            tlist.append(t)
        return np.array(nlist),np.array(tlist)
        #return points
    
    def fit_circle(self,points): #indep func
        # cite  https://blog.csdn.net/jiangjjp2812/article/details/106937333/
        if self.a is None:
            self.fit_surface(points)
        M,_ = self.reflect(points)
        num,dim=points.shape
        L1 = np.ones((num,1))
        A = np.linalg.inv(M.T.dot(M)).dot(M.T).dot(L1).reshape(3,-1)
        
        """ 
        #v1
        t0 = time.time()
        B = []
        for i in range(num):
            for j in range(i+1,num):
                B.append(M[j]-M[i])
        B = np.array(B)
        print("e",t0-time.time())
        L2 = []
        for i in range(num):
            for j in range(i+1,num):
                temp =np.sum(M[j]**2-M[i]**2)/2
                L2.append(temp)
        L2 = np.array(L2)
        L2 = L2.reshape(L2.shape[0],1)
        """
        """
        #v2
        Mshape = M.shape[0]
        tril_idx = np.triu(np.ones((Mshape,Mshape))-np.eye(Mshape)).flatten().nonzero()
        M_minus = M.reshape(Mshape,1,3)-M.reshape(1,Mshape,3)
        B = []
        for i in range(num):
            for j in range(i+1,num):
                B.append(M_minus[j,i])
        B = np.array(B)
        M_sq = (M**2).sum(-1)
        
        Msq_minus = (M_sq.reshape(Mshape,1)-M_sq.reshape(1,Mshape))/2
        L2 = []
        for i in range(len(Msq_minus)):
            for j in range(i+1,len(Msq_minus)):
                L2.append(Msq_minus[j][i])
        L2 = np.array(L2)
        L2 = L2.reshape(L2.shape[0],1)
        """
        #v3
        Mshape = M.shape[0]
        tril_idx = np.triu(np.ones((Mshape,Mshape))-np.eye(Mshape)).flatten().nonzero()
        M_minus = M.reshape(Mshape,1,3)-M.reshape(1,Mshape,3)
        B = -M_minus.reshape(-1,3)[tril_idx]
        #print("dif",np.abs(B-B1).sum())
        
        M_sq = (M**2).sum(-1)
        Msq_minus = (M_sq.reshape(Mshape,1)-M_sq.reshape(1,Mshape))/2
        L2 = - Msq_minus.flatten()[tril_idx]
        L2 = L2.reshape(L2.shape[0],1)
        ####

        
        D=np.zeros((4,4))
        D[:3,:3] = B.T.dot(B)
        D[3:4,:3] = A.T
        D[:3,3:4] = A 
        
        temp = B.T.dot(L2)
        L3 = np.zeros((temp.shape[0]+1,1))
        L3[:-1,0:1] = temp
        C = np.linalg.inv(D).dot(L3)
        C = C[0:3]

        C = self.reflect_one_point(C)[0]
        r = 0
        for x in M:
            r += np.linalg.norm(x-C.T[0])
        r /= num
        #print(M.max(axis=0),M.min(axis=0),C.T[0],r)
        return [C.T[0],r]


class Eye_reconstructor():
    def __init__(self):
        self.eyeidx = self.get_orbit_idx()
        self.one_eye_points = om.read_polymesh("networks/v0/pose/eyes/one-eye.obj").points() # copy when use
        eyemesh = om.read_polymesh("networks/v0/pose/eyes/template_eyes.obj")
        self.eyeface = eyemesh.face_vertex_indices()

    def get_orbit_idx(self):
        mesh = om.read_polymesh("networks/v0/pose/eyes/color-eye.ply",vertex_color=True)
        colors = mesh.vertex_colors()
        eye_idx = []
        for i,color in enumerate(colors):
            if color[0]<0.3 and color[1]<0.3:
                    eye_idx.append(i)
        return eye_idx

    def get_ball_info(self,points): 
        eye_max = points.max(axis=0)
        eye_min = points.min(axis=0)
        eye_c = (eye_max+eye_min)/2
        eye_r = abs(eye_max[0]-eye_min[0])/2
        return eye_c,eye_r
    
    def generate_one_eye(self,mesh_points,rC,rR):
        orbit = mesh_points[self.eyeidx]
        surface = Surface()
        circle = surface.fit_circle(orbit)
        vec = np.array([surface.a,surface.b,surface.c]) 
        if surface.c<0:
            vec = -vec
        vec = vec/np.linalg.norm(vec) #norm vectore
        rotation = R.align_vectors(np.reshape(vec,(1,-1)),np.array([[0,0,1]]) )
        rotation = rotation[0].as_matrix()
        # print(rotation)
        bC = circle[0] - vec*rC*circle[1]
        bR= circle[1]*rR
        template_points = self.one_eye_points.copy()
        template_points = template_points.dot(rotation.T)
        #faces = template_eye.face_vertex_indices()  
        tc,tr = self.get_ball_info(template_points)
        template_points = template_points/tr*bR
        tc,tr = self.get_ball_info(template_points)  #now tr==bR
        template_points = template_points-tc+bC
        return template_points        

    def reconstruct(self,meshpoints,rC=None,rR=None):
        meshpoints = meshpoints.detach().cpu().numpy()
        eye1 = self.generate_one_eye(meshpoints,rC,rR)
        eye2 = eye1.copy() # utilze symmetric
        eye2[:,0] = -eye2[:,0]
        #eye2 = generate_one_eye(eye2_idx)
        eyes = np.concatenate([eye1,eye2],axis=0)
        #reconstruct
        newmesh = om.PolyMesh(points=eyes,face_vertex_indices=self.eyeface)
        return newmesh

#reconstruct("t2",rC,rR)

if __name__=='__main__':
    pass
    """
    #test1
    circle = om.read_polymesh("circle.obj")
    points = circle.points()
    test(points)
    eye1_points, eye2_points,_,_ = get_circle()
    test(eye1_points,"eye1_surface.obj")
    test(eye2_points,"eye2_surface.obj")
    """
    """
    #test2
    eye1_points, eye2_points,_,_ = get_circle()
    test2(eye1_points,"eye1_point_circle1.obj")
    """
    #test4()
