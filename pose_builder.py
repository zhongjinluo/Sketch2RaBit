import numpy as np
import pickle
import pdb
import openmesh as om
import trimesh

class PoseBuilder():
  def __init__(self):
    dataroot = "networks/v0/pose/"
    self.clusterdic = np.load(dataroot + 'clusterdic.npy', allow_pickle=True).item()
    self.index2cluster = {}
    for key in self.clusterdic.keys():
    	val = self.clusterdic[key]
    	self.index2cluster[val] = key
    self.joint2index = np.load(dataroot + 'joint2index.npy', allow_pickle=True).item()
    ktree_table = np.load(dataroot + 'ktree_table.npy', allow_pickle=True).item()
    joint_order = np.load(dataroot + "pose_order.npy")
    self.weightMatrix = np.load(dataroot + 'weightMatrix.npy', allow_pickle=True)
    self.pose_database = np.load(dataroot + "pose_gt.npy",allow_pickle=True).item()
    self.ktree_table = np.ones(24)*-1
    name2index = {}
    for i in range(1,24):
        self.ktree_table[i]=ktree_table[i][1]
        name2index[ktree_table[i][0]]=i
    reorder_index = np.zeros(24)
    for i,jointname in enumerate(joint_order):
        if jointname in name2index:
            reorder_index[name2index[jointname]]=i
        else:
            reorder_index[0]=2
    self.reorder_index = np.array(reorder_index).astype(int)
    self.weights = self.weightMatrix
    self.parent = self.ktree_table

    self.pose_shape = [24, 3]
    self.trans_shape = [3]

    self.pose = np.zeros(self.pose_shape)
    self.trans = np.zeros(self.trans_shape)

    self.verts = None
    self.J = None
    self.R = None

  def set_params(self, T_vertices, pose, k="temp"):
      # self.beta = beta
      self.pose = pose[self.reorder_index]
      # print(self.reorder_index)
      return self.update(T_vertices, k=k)

  def update(self, T_vertices, k="temp"):
    """
    Called automatically when parameters are updated.

    """
    v_shaped = T_vertices.reshape(-1)

    # joints location
    # self.J = self.J_regressor.dot(v_shaped)
    # we use a new way to evaluate J
    self.J = []
    for i in range(len(self.index2cluster)):
    	key = self.index2cluster[i]
    	if key =='RootNode':
    		self.J.append(np.array([0,0,0]))
    		continue
    	index_list = self.joint2index[key]
    	index_val = []
    	for index in index_list:
    		index_val.append(T_vertices[index])
    	index_val = np.array(index_val)
    	maxval = index_val.max(0)
    	minval = index_val.min(0)
    	self.J.append((maxval+minval)*1.0/2)
    self.J = np.array(self.J)

    pose_cube = self.pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    self.R = self.rodrigues(pose_cube)
    I_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      (self.R.shape[0]-1, 3, 3)
    )
    lrotmin = (self.R[1:] - I_cube).ravel()

    v_posed = v_shaped.reshape(-1,3)
    # world transformation of each joint
    G = np.empty((self.ktree_table.shape[0], 4, 4))
    G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
    for i in range(1, self.ktree_table.shape[0]):
      G[i] = G[int(self.parent[i])].dot(
        self.with_zeros(
          np.hstack(
            [self.R[i],((self.J[i, :]-self.J[int(self.parent[i]),:]).reshape([3,1]))]
          )
        )
      )
    # remove the transformation due to the rest pose
    G0 = G
    G = G - self.pack(
      np.matmul(
        G,
        np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
        )
      )
    # transformation of each vertex
    T = np.tensordot(self.weights, G, axes=[[1], [0]])
    rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
    v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
    self.verts = v + self.trans.reshape([1, 3])

    posed_vertices = self.verts.reshape(-1, 3)
    skeleton = []
    for i in range(len(self.index2cluster)):
    	key = self.index2cluster[i]
    	if key =='RootNode':
    		skeleton.append(np.array([0,0,0]))
    		continue
    	index_list = self.joint2index[key]
    	index_val = []
    	for index in index_list:
    		index_val.append(posed_vertices[index])
    	index_val = np.array(index_val)
    	maxval = index_val.max(0)
    	minval = index_val.min(0)
    	skeleton.append((maxval+minval)*1.0/2)
    skeleton = np.array(skeleton)

    return posed_vertices, skeleton
  

  def rodrigues(self, r):
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(np.float64).tiny)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R

  def with_zeros(self, x):
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

  def pack(self, x):
    return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

  def build(self, T_vertices, k):
    pose = self.pose_database[k].reshape(24,3)
    # posed_vertices, skeleton = self.set_params(T_vertices=T_vertices, pose=pose, k=k)
    print(pose.max())
    return pose


if __name__ == '__main__':
  import trimesh
  pb = PoseBuilder()
  coeff2 = np.load("networks/v0/pose/pose_gt.npy",allow_pickle=True).item()
  for k in list(coeff2.keys()):
    k = "20210719_design_猫_1"
    pose = coeff2[k].reshape(24,3)
    m = om.read_polymesh("networks/v0/pose/m.OBJ")
    vertices, skeleton = pb.set_params(T_vertices=m.points(), pose=pose, k=k)
    mesh = trimesh.Trimesh(vertices=vertices, faces=m.face_vertex_indices(), process=False)
    mesh.export(k + ".obj")
    print(k)
    break


