# -*- Python -*-
# Jiao Lin <linjiao@caltech.edu>

import numpy as np

def computePrincipalVectors(data):
    # position of max intensity
    I = data[:, 4]
    maxpos = np.argmax(I)
    print data[maxpos]
    h0,k0,l0 = data[maxpos][:3]
    # find points around the max intensity
    diff = data[:, :3] - np.array((h0, k0, l0))
    l2 = np.sum(diff**2, -1)
    sortindexes = np.argsort(l2)
    diff = diff[sortindexes]
    u = diff[1]
    print "u=", u
    for i, d in enumerate(diff[2:]): # skip the zero
        ca = cosa(d, u)
        print ca, d
        if ca > 0.8:
            v = d; break
        continue
    print "v=", v
    for j, d in enumerate(diff[i+2:]): 
        ca1 = cosa(d,u); ca2 = cosa(d, v)
        # print ca1, ca2, d
        if ca1 > 0.8 and ca2 > 0.8:
            print ca1, ca2, d
            w = d; break
        continue
    print "w=", w
    print u,v,w
    M = np.array([u, v, w]).T
    IM = np.linalg.inv(M)
    for d in diff[:i+2+j+5]:
        v = np.dot(IM, d)
        iv = np.round(v)
        print iv, v-iv
        continue
    return u, v, w


def cosa(v1, v2):
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    return np.linalg.norm( np.cross(v1, v2) )


def compute_dE(data):
    E = data[:, 3]
    E1 = np.unique(E)
    if len(E1)%2 == 1:
        median = np.median(E1)
    else:
        median = np.median(E1[:-1])
    index = np.where(E1==median)[0][0]
    dE1 = E1[index] - E1[index-1] 
    dE2 = E1[index+1] - E1[index] 
    assert (dE1 - dE2) < 1e-6*dE1
    return dE1


def gauss(x, sigma):
    t = x/sigma
    return np.exp(-t*t/2.) / (np.sqrt(2*np.pi)*sigma)


def box(x):
    y = np.zeros(x.size, dtype='double')
    y[np.abs(x) < 1] = 1
    return y


GW = 1./1.2
class SkewedGaussian:
    
    def __init__(self, u, v, w):
        M = np.array([u, v, w]).T
        self.IM = np.linalg.inv(M)
        return
    
    
    def __call__(self, h, k, l):
        """h,k,l are all arrays of same size
        """
        hkl = np.vstack((h,k,l)) # 3Xn
        v = np.dot(self.IM, hkl) # 3Xn
        # v /= 2
        # v *= GW
        # v *= 1.5 # XXX check this
        # v *= 1.85 # XXX check this
        # v *= 2 # XXX check this
        # v *= 2.355
        return gauss(v[0], GW) * gauss(v[1], GW) * gauss(v[2], GW) 


class FunctionFromPointCloud:

    def __init__(self, path):
        data = np.fromfile(path)
        data.shape = -1,6
        self.data = data
        self.uvw = u, v, w = computePrincipalVectors(data)
        self.dE = compute_dE(data) # * 100
        return
    
    
    def __call__(self, h, k, l, E):
        "h, k, l, E are all arrays of same size"
        N = h.size
        I = np.zeros(N, dtype='double')
        u,v,w = self.uvw
        dE = self.dE
        for h0, k0, l0, E0, I0, dI0 in self.data:
            sg = SkewedGaussian(u,v,w)
            h1 = h-h0
            k1 = k-k0
            l1 = l-l0
            I1 = sg(h1, k1, l1)
            # I1 *= gauss((E-E0)/dE)
            # I1 *= gauss((E-E0)/dE * .5)
            I1 *= gauss((E-E0)/dE, GW)
            # I1 *= gauss((E-E0)/dE * 1.5)
            # I1 *= gauss((E-E0)/dE * 1.85)
            # I1 *= gauss((E-E0)/dE * 2)
            # I1 *= gauss((E-E0)/dE * 2.355)
            I += I1 * I0
            continue
        return I


# sample the reconstructed point cloud
def sample(pc, h0, k0, l0, E0, dh, dk, dl, dE, N):
    """pc: pointcloud path
    h0,k0,l0, E0: point of interest
    N: number of points in the sample
    """
    f = FunctionFromPointCloud(pc)
    # center at hkl=-5.333, -2.667, 2.667, E=37
    # h0, k0, l0 = -(5+1./3), -(2+2./3), 2+2./3
    # E0 = 37
    
    # N = 400000
    # N = 50000
    # N = 1000000
    
    xmin, xmax = h0-dh, h0+dh
    ymin, ymax = k0-dk, k0+dk
    zmin, zmax = l0-dl, l0+dl
    umin, umax = E0-dE, E0+dE
    x = np.random.random(N) * (xmax-xmin) + xmin
    y = np.random.random(N) * (ymax-ymin) + ymin
    z = np.random.random(N) * (zmax-zmin) + zmin
    u = np.random.random(N) * (umax-umin) + umin
    I = f(x, y, z, u)
    return x,y,z,u, I


def project_xE(
    pc, hkl0, hkl_dir, x0, E0,
    dh=.25, dk=.25, dl=.25, dE=3,
    N=1000000,
    out = "proj_xE.h5",
    ):
    """compute slice in the x,E plane
    
    For example, we want plot of horizontal axis of hkl=x,1,x
    and vertical axis of E.
    Then hkl0 = 0,1,0, and hkl_dir = 1,0,1
    
    x0, E0: point of interest. for example, we are interested in
            hkl=2,1,2, E=30. that means x0=2, E0=30
    
    """
    hkl0 = np.array(hkl0)
    hkl_dir = np.array(hkl_dir)
    h0,k0,l0 = hkl0+x0*hkl_dir
    x,y,z,u,I = sample(pc, h0, k0, l0, E0, dh, dk, dl, dE, N=N)
    hkl = np.vstack((x,y,z)).T
    diff = hkl - hkl0
    len = np.linalg.norm(hkl_dir)
    e = hkl_dir / len
    l = np.dot(hkl, e)
    xx = l/len
    data = np.vstack((xx,u)).T
    h, edges = np.histogramdd(data, bins=(50,50), weights=I)
    h/=N
    xe, Ee = edges
    h/=(xe[1]-xe[0])*(Ee[1]-Ee[0])
    import histogram as H, histogram.hdf as hh
    axes = [H.axis('x', boundaries=xe), 
            H.axis('E', boundaries=Ee, unit='meV')]
    hist = H.histogram('Resolution Slice', axes, data=h)
    hh.dump(hist, out)
    return


def slice_xE(pc, hkl0, hkl_dir, x0, E0, xrange, Erange):
    """compute slice in the x,E plane
    
    For example, we want plot of horizontal axis of hkl=x,1,x
    and vertical axis of E.
    Then hkl0 = 0,1,0, and hkl_dir = 1,0,1
    
    x0, E0: point of interest. for example, we are interested in
            hkl=2,1,2, E=30. that means x0=2, E0=30
    
    xrange: 
    Erange:
    """
    f = FunctionFromPointCloud(pc)
    # h0, k0, l0 = -(5+1./3), -(2+2./3), 2+2./3
    hkl0 = np.array(hkl0)
    hkl_dir = np.array(hkl_dir)
    
    x1 = np.arange(*xrange)
    u1 = np.arange(*Erange)
    hkl = x1[:, np.newaxis]*hkl_dir + hkl0
    h1,k1,l1 = hkl.T
    x = np.repeat(h1, u1.size)
    y = np.repeat(k1, u1.size)
    z = np.repeat(l1, u1.size)
    u = np.tile(u1, h1.size)
    I = f(x, y, z, u)
    import histogram as H, histogram.hdf as hh
    axes = [('x', x1), ('E', u1, 'meV')]
    I.shape = x1.size, u1.size
    h = H.histogram('Resolution Slice', axes, data=I)
    hh.dump(h, "res_xE.h5")
    return
    


def slice_hk():
    f = FunctionFromPointCloud('pointcloud')
    # center at hkl=-5.333, -2.667, 2.667, E=37
    h0, k0, l0 = -(5+1./3), -(2+2./3), 2+2./3
    E0 = 37
    
    x1 = np.arange(-0.1, 0.1, 0.001)
    y1 = np.arange(-0.1, 0.1, 0.001)
    x = np.repeat(x1, y1.size) + h0
    y = np.tile(y1, x1.size) + k0
    z = np.repeat(l0, x.size)
    u = np.repeat(E0, x.size)
    I = f(x, y, z, u)
    # I = np.log(I+1)
    import histogram as H, histogram.hdf as hh
    axes = [('h', x1), ('k', y1)]
    I.shape = x1.size, y1.size
    h = H.histogram('Resolution Slice', axes, data=I)
    hh.dump(h, "reshk.h5")
    return


def slice_hl():
    f = FunctionFromPointCloud('pointcloud')
    # center at hkl=-5.333, -2.667, 2.667, E=37
    h0, k0, l0 = -(5+1./3), -(2+2./3), 2+2./3
    E0 = 37
    
    x1 = np.arange(-0.1, 0.1, 0.001)
    z1 = np.arange(-0.1, 0.1, 0.001)
    x = np.repeat(x1, z1.size) + h0
    z = np.tile(z1, x1.size) + l0
    y = np.repeat(k0, x.size)
    u = np.repeat(E0, x.size)
    I = f(x, y, z, u)
    # I = np.log(I+1)
    import histogram as H, histogram.hdf as hh
    axes = [('h', x1), ('l', z1)]
    I.shape = x1.size, z1.size
    h = H.histogram('Resolution Slice', axes, data=I)
    hh.dump(h, "reshl.h5")
    return


def slice_kl():
    f = FunctionFromPointCloud('pointcloud')
    # center at hkl=-5.333, -2.667, 2.667, E=37
    h0, k0, l0 = -(5+1./3), -(2+2./3), 2+2./3
    E0 = 37
    
    y1 = np.arange(-0.15, 0.15, 0.001)
    z1 = np.arange(-0.15, 0.15, 0.001)
    y = np.repeat(y1, z1.size) + k0
    z = np.tile(z1, y1.size) + l0
    x = np.repeat(h0, z.size)
    u = np.repeat(E0, z.size)
    I = f(x, y, z, u)
    import histogram as H, histogram.hdf as hh
    axes = [('k', y1), ('l', z1)]
    I.shape = y1.size, z1.size
    h = H.histogram('Resolution Slice', axes, data=I)
    hh.dump(h, "reskl.h5")
    return


def slice_hE():
    f = FunctionFromPointCloud('pointcloud')
    # center at hkl=-5.333, -2.667, 2.667, E=37
    h0, k0, l0 = -(5+1./3), -(2+2./3), 2+2./3
    E0 = 37
    
    x1 = np.arange(-0.15, 0.15, 0.001)
    u1 = np.arange(-4, 4, 0.05)
    x = np.repeat(x1, u1.size) + h0
    u = np.tile(u1, x1.size) + E0
    y = np.repeat(k0, x.size)
    z = np.repeat(l0, x.size)
    I = f(x, y, z, u)
    import histogram as H, histogram.hdf as hh
    axes = [('h', x1), ('E', u1, 'meV')]
    I.shape = x1.size, u1.size
    h = H.histogram('Resolution Slice', axes, data=I)
    hh.dump(h, "reshE.h5")
    return


def slice_kE():
    f = FunctionFromPointCloud('pointcloud')
    # center at hkl=-5.333, -2.667, 2.667, E=37
    h0, k0, l0 = -(5+1./3), -(2+2./3), 2+2./3
    E0 = 37
    
    y1 = np.arange(-0.15, 0.15, 0.001)
    u1 = np.arange(-4, 4, 0.05)
    y = np.repeat(y1, u1.size) + k0
    u = np.tile(u1, y1.size) + E0
    x = np.repeat(h0, u.size)
    z = np.repeat(l0, u.size)
    I = f(x, y, z, u)
    import histogram as H, histogram.hdf as hh
    axes = [('k', y1), ('E', u1, 'meV')]
    I.shape = y1.size, u1.size
    h = H.histogram('Resolution Slice', axes, data=I)
    hh.dump(h, "reskE.h5")
    return


def slice_lE():
    f = FunctionFromPointCloud('pointcloud')
    # center at hkl=-5.333, -2.667, 2.667, E=37
    h0, k0, l0 = -(5+1./3), -(2+2./3), 2+2./3
    E0 = 37
    
    z1 = np.arange(-0.15, 0.15, 0.001)
    u1 = np.arange(-4, 4, 0.05)
    z = np.repeat(z1, u1.size) + l0
    u = np.tile(u1, z1.size) + E0
    x = np.repeat(h0, u.size)
    y = np.repeat(k0, u.size)
    I = f(x, y, z, u)
    import histogram as H, histogram.hdf as hh
    axes = [('l', z1), ('E', u1, 'meV')]
    I.shape = z1.size, u1.size
    h = H.histogram('Resolution Slice', axes, data=I)
    hh.dump(h, "reslE.h5")
    return


def test1():
    f = FunctionFromPointCloud('pointcloud')
    # center at hkl=-5.333, -2.667, 2.667, E=37
    N = 100
    h = np.repeat(-5.333, N)
    k = np.repeat(-2.667, N)
    l = np.repeat(2.667, N)
    Emin, Emax = 0, 100
    dE = (Emax-Emin)/N
    E = np.arange(Emin, Emax-dE/2., dE)
    I = f(h, k, l, E)
    print E
    print I
    import pylab
    pylab.plot(E,I)
    pylab.show()
    return


def test2():
    f = FunctionFromPointCloud('pointcloud')
    # center at hkl=-5.333, -2.667, 2.667, E=37
    h0, k0, l0 = -(5+1./3), -(2+2./3), 2+2./3
    E0 = 37
    
    Nx, Ny = 80, 100
    
    xmin, xmax = h0-0.5, h0+0.5
    xmin, xmax = h0-0.2, h0+0.2
    dx = (xmax-xmin)/Nx
    x = np.arange(xmin, xmax-dx/2, dx)
    
    ymin, ymax = k0-0.5, k0+0.5
    ymin, ymax = k0-0.2, k0+0.2
    dy = (ymax-ymin)/Ny
    y = np.arange(ymin, ymax-dy/2, dy)

    x1 = np.repeat(x, Ny)
    y1 = np.tile(y, Nx)
    
    h = x1
    k = y1
    l = np.repeat(l0, Nx*Ny)
    E = np.repeat(E0, Nx*Ny)
    
    I = f(h, k, l, E)
    I.shape = Nx, Ny
    
    axes = [
        ('h', x),
        ('k', y),
        ]
    import histogram as H, histogram.hdf as hh
    h = H.histogram('Ihk', axes, data=I)
    hh.dump(h, "Ihk.h5")
    return


def main():
    test2()
    return


if __name__ == '__main__':
    main()


# End of file
