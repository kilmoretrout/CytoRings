import numpy as np
from scipy.interpolate import UnivariateSpline

from scipy.integrate import odeint
import copy

import matplotlib.pyplot as plt
from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D

import trimesh

from pyefd import elliptic_fourier_descriptors

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def differentiator(f, N = 7):
    if N == 7:
        ret = []
        for k in range(3, len(f) - 3):
            ret.append((39*(f[k + 1] - f[k - 1]) + 12*(f[k + 2] - f[k - 2]) - 5*(f[k + 3] - f[k - 3])) / (96))

        return ret
    elif N == 11:
        ret = []
        for k in range(5, len(f) - 5):
            ret.append((322*(f[k + 1] - f[k - 1]) + 256*(f[k + 2] - f[k - 2]) + 39*(f[k + 3] - f[k - 3]) - 32*(f[k + 4] - f[k - 4]) - 11*(f[k + 5] - f[k-5])) / (1536))

        return ret

class PointCloud(object):
    def __init__(self, N = 1024, Nk = 3):
        self.theta = np.linspace(0., 2*np.pi, 72)

        # number of sampling points to take at each
        # axial slice
        self.N = N

        # number of harmonics in the axial fits
        self.Nk = Nk
        self.xy_phi = None

        return

    def cone(self, noise_sigma = 0.1, n_frames = None):
        if n_frames is None:
            self.n_frames = int(np.round(np.random.uniform(60, 110)))
        else:
            self.n_frames = n_frames

        # real time in seconds for microscopy data
        self.t = np.linspace(0., 1., self.n_frames)
        r0 = 100

        r = 125. - 100*self.t

        # only for the cone object
        self.real_speed = np.abs(r[0] - r[1])
        y_displacement = np.array(range(len(self.t))) * self.real_speed / 1.5

        self.t *= 2.7 * self.n_frames

        theta = np.linspace(-np.pi, np.pi, 72)

        self.X = []
        self.X_theta = []

        for k in range(len(self.t)):
            x = []
            y = []

            x_ = r[k] * np.cos(theta)
            y_ = r[k] * np.sin(theta) - y_displacement[k]

            plt.plot(x_, y_, color = 'k')

            for j in range(100):
                x.extend(list(x_ + np.random.normal(0, noise_sigma, 72)))
                y.extend(list(y_ + np.random.normal(0, noise_sigma, 72)))

            X_ = np.vstack([np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)]).T
            theta_ = np.arctan2(X_[:, 1], X_[:, 0])

            self.X.append(X_)
            self.X_theta.append(theta_)

        plt.show()

    def elliptical_cone(self, starting_major_axis = 150., noise = 2., perturb_centroid = False):
        # define the numbe of frames and the time parameterization
        self.n_frames = int(np.round(np.random.uniform(60, 110)))
        # real time in seconds for microscopy data
        self.t = np.array(range(self.n_frames)) * 2.7

        alpha0 = starting_major_axis
        beta0 = np.random.uniform(0.65, 0.9)*alpha0

        alpha = alpha0 - self.t*alpha0 / np.max(self.t)
        beta = beta0 - self.t*beta0 / np.max(self.t)

        self.alpha0 = alpha0
        self.beta0 = beta0

        X = []
        X_theta = []

        for k in range(len(self.t)):
            x = []
            y = []

            x_ = alpha[k] * np.cos(self.theta)
            y_ = beta[k] * np.sin(self.theta)

            for j in range(100):
                x.extend(list(x_ + np.random.normal(0, noise, 72)))
                y.extend(list(y_ + np.random.normal(0, noise, 72)))

            X_ = np.vstack([np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)]).T
            theta_ = np.arctan2(X_[:, 1], X_[:, 0])

            X.append(X_)
            X_theta.append(theta_)

        self.X = np.array(X)
        self.X_theta = np.array(X_theta)

        X = np.vstack(X)

    def load_array(self, xy):
        self.X = xy
        self.X_theta = np.zeros(xy.shape[:2])

        for k in range(self.X.shape[0]):
            xy_ = self.X[k]
            m = np.mean(xy_, axis = 0)

            self.X_theta[k] = np.arctan2(xy_[:,1] - m[1], xy_[:,0] - m[0])

        self.t = np.array(range(self.X.shape[0]))*2.7
        self.n_frames = len(self.t)

    def load_file(self, ifile, trim = 100, n_frames = 100):
        mesh = trimesh.load(ifile)

        mesh.vertices -= mesh.center_mass
        mesh.vertices[:, -1] -= np.min(mesh.vertices[:, -1])

        bins = np.linspace(np.percentile(mesh.vertices[:,-1], 10), np.percentile(mesh.vertices[:,-1], 90), n_frames + 1)

        # real time in seconds for microscopy data
        self.t = []

        X = []
        X_theta = []

        indices = np.digitize(mesh.vertices[:,-1], bins)

        for k in range(1,len(bins) - 1):
            ix = list(np.where(indices == k)[0])

            if len(ix) > 1000:

                X_ = mesh.vertices[ix,:2]

                theta_ = np.arctan2(X_[:,1], X_[:,0])

                #plt.plot(X_[np.argsort(theta_), 0], X_[np.argsort(theta_), 1])
                #plt.show()

                X.append(X_)
                X_theta.append(theta_)

                self.t.append(bins[k])

        self.X = X


        self.X_theta = X_theta
        self.t = np.array(self.t)

        self.n_frames = len(self.t)

    def solve_axial_slices(self, s = None, smooth = True):
        co = []
        xy = []
        xy_real = []

        for k in range(len(self.X)):
            xy_ = self.X[k]
            theta_ = self.X_theta[k]

            ix = list(set([np.argmin(np.abs(theta_ - u)) for u in np.linspace(-np.pi, np.pi, self.N)]))

            x_co = solve_harmonic_series(xy_[ix, 0], theta_[ix], N = self.Nk)
            y_co = solve_harmonic_series(xy_[ix, 1], theta_[ix], N = self.Nk)

            xy_real.append(xy_[ix])

            co.append(np.vstack([x_co, y_co]))

        self.co = np.array(co)
        self.xy_real = np.array(xy_real)

        if smooth:
            self.co_f = dict()
            self.co_f['x'] = []
            self.co_f['y'] = []

            for k in range(self.co.shape[-1]):
                # x
                f = UnivariateSpline(self.t, self.co[:,0,k], s = s)
                # replace
                self.co[:,0,k] = f(self.t)
                self.co_f['x'].append(f)

                # y
                f = UnivariateSpline(self.t, self.co[:, 1, k], s = s)
                # replace
                self.co[:,1,k] = f(self.t)
                self.co_f['y'].append(f)

        x_norm = []

        norm_y = np.sqrt(self.co[:,1,2]**2 + self.co[:,1,1]**2)
        phase_y = np.arctan2(self.co[:,1,2], self.co[:,1,1])

        norm_x = np.sqrt(self.co[:, 0, 2] ** 2 + self.co[:, 0, 1] ** 2)
        phase_x = np.arctan2(self.co[:, 0, 2], self.co[:, 0, 1])

        for k in range(len(self.X)):
            x_co = self.co[k,0,:]
            y_co = self.co[k,1,:]

            x_new = compute_harmonic(self.theta, x_co, N=self.Nk)
            y_new = compute_harmonic(self.theta, y_co, N=self.Nk)

            xy.append(np.array([x_new, y_new]).T)

        self.xy = np.array(xy, dtype = np.float32)

    def specify_derivatives(self, t, co):
        self.co = co
        self.t = t
        self.n_frames = len(t)

        self.co_f = dict()
        self.co_f['x'] = []
        self.co_f['y'] = []

        xy = []

        for k in range(self.co.shape[0]):
            x_co = self.co[k,0,:]
            y_co = self.co[k,1,:]

            x_new = compute_harmonic(self.theta, x_co, N=self.Nk)
            y_new = compute_harmonic(self.theta, y_co, N=self.Nk)

            xy.append(np.array([x_new, y_new]).T)

        self.xy = np.array(xy, dtype = np.float32)

        for k in range(self.co.shape[-1]):
            # x
            f = UnivariateSpline(self.t, self.co[:, 0, k], s = 10e-6)
            # replace
            self.co[:, 0, k] = f(self.t)
            self.co_f['x'].append(f)

            # y
            f = UnivariateSpline(self.t, self.co[:, 1, k], s = 10e-6)
            # replace
            self.co[:, 1, k] = f(self.t)
            self.co_f['y'].append(f)

    def dtheta_vec(self, phi):
        _ = []

        for ix in range(1, self.Nk + 1):
            _.append(-np.sin(phi * ix) * ix)

        for ix in range(1, self.Nk + 1):
            _.append(np.cos(phi * ix) * ix)

        return np.array(_)

    def dt_vec(self, phi):
        _ = []
        _.append(1.)

        for ix in range(1, self.Nk + 1):
            _.append(np.cos(phi * ix))

        for ix in range(1, self.Nk + 1):
            _.append(np.sin(phi * ix))

        return np.array(_)

    def model(self, phi, t):
        phi = phi[0]

        co_x = np.array([f(t) for f in self.co_f['x']])
        co_y = np.array([f(t) for f in self.co_f['y']])

        co_x_dt = np.array([f.derivative(1)(t) for f in self.co_f['x']])
        co_y_dt = np.array([f.derivative(1)(t) for f in self.co_f['y']])

        dtheta_vec = self.dtheta_vec(phi)
        dt_vec = self.dt_vec(phi)

        dx_dtheta = dtheta_vec.dot(co_x[1:])
        dy_dtheta = dtheta_vec.dot(co_y[1:])
        x_dt = dt_vec.dot(co_x_dt)
        y_dt = dt_vec.dot(co_y_dt)

        return (-dx_dtheta*x_dt - dy_dtheta*y_dt) / (dx_dtheta**2 + dy_dtheta**2)

    def model_arc_length(self, theta, t = 0.):
        dtheta_vec = self.dtheta_vec(theta)

        co_x = np.array([f(t) for f in self.co_f['x']])
        co_y = np.array([f(t) for f in self.co_f['y']])

        dx_dtheta = dtheta_vec.dot(co_x[1:])
        dy_dtheta = dtheta_vec.dot(co_y[1:])

        return np.sqrt(dx_dtheta**2 + dy_dtheta**2)

    def model_elliptical_cone(self, phi, t):
        return

    def solve_phi(self):
        self.phi = np.zeros((self.n_frames, len(self.theta)))
        self.xy_phi = np.zeros((len(self.t), len(self.theta), 2))

        for i in range(len(self.theta)):
            self.phi[:,i] = odeint(self.model, self.theta[i], self.t).flatten()

        for i in range(self.phi.shape[0]):
            A = get_A(self.phi[i], N = self.Nk)

            self.xy_phi[i,:,0] = A.dot(self.co[i,0,:].T)
            self.xy_phi[i,:,1] = A.dot(self.co[i,1,:].T)

    def compute_speeds(self):
        # let's look at speed as estimated by a differentiator
        dx = np.array(differentiator(self.xy[:, :, 0]))
        dy = np.array(differentiator(self.xy[:, :, 1]))

        speed = np.sqrt(dx**2 + dy**2)

        if self.xy_phi is not None:
            dx = np.array(differentiator(self.xy_phi[:, :, 0]))
            dy = np.array(differentiator(self.xy_phi[:, :, 1]))

            speed_phi = np.sqrt(dx**2 + dy**2)

            return speed.T, speed_phi.T
        else:
            return speed.T

    def plot(self):
        bounds = (np.min(self.xy), np.max(self.xy))

        A_prime = get_A_prime(self.theta, N = self.Nk)

        dx = []
        dy = []

        for k in range(len(self.co)):
            dx.append(A_prime.dot(self.co[k,0,:]))
            dy.append(A_prime.dot(self.co[k,1,:]))

        dx = np.array(dx, dtype = np.float32)
        dy = np.array(dy, dtype = np.float32)

        norm = np.sqrt(dx**2 + dy**2)

        dx /= norm
        dy /= norm

        print(self.xy.dtype, dx.dtype)

        k = list(range(0, len(self.xy) - 10, 5))
        print(dy[k,:].flatten()*np.tile(np.cos(self.theta), (len(k),1)).flatten())
        print(dx[k,:].flatten()*np.tile(np.sin(self.theta), (len(k),1)).flatten())

        plt.quiver(self.xy[k,:,0].flatten(), self.xy[k,:,1].flatten(), -dy[k,:].flatten(), dx[k,:].flatten(),
                   np.arccos((dy[k,:].flatten()*np.tile(np.cos(self.theta), (len(k),1)).flatten()
                             - dx[k,:].flatten()*np.tile(np.sin(self.theta), (len(k),1)).flatten())), scale_units = 'inches', scale=4, color='g')

        plt.colorbar()

        for j in range(0, 72):
            plt.plot(self.xy_phi[:,j,0], self.xy_phi[:,j,1], color = 'b')

        plt.show()
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')

        for k in range(len(self.xy)):
            ax.plot(self.xy[k,:,0], self.xy[k,:,1], [self.t[k] for u in range(len(self.theta))], color = 'k')

        for j in range(0, self.xy_phi.shape[1]):
            ax.plot(self.xy_phi[:,j,0], self.xy_phi[:,j,1], self.t, color = 'b')

        ax = fig.add_subplot(122, projection='3d')

        for k in range(len(self.xy)):
            ax.plot(self.xy[k, :, 0], self.xy[k, :, 1], [self.t[k] for u in range(len(self.theta))], color='k')

        for j in range(0, self.xy_phi.shape[1]):
            ax.plot(self.xy[:, j, 0], self.xy[:, j, 1], self.t, color='b')

        plt.show()
        plt.close()

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize = (6,12))

        axes[0, 0].set_title('original (no phase function)')

        axes[0, 0].set_xlim(*bounds)
        axes[0, 1].set_xlim(*bounds)

        axes[0, 0].set_ylim(*bounds)
        axes[0, 1].set_ylim(*bounds)

        ix = list(range(0, self.xy.shape[1], 3))

        for k in ix:
            axes[0, 0].plot(self.xy[:, k, 0], self.xy[:, k, 1], color = 'k')

        #for k in range(self.xy_phi.shape[0]):
        #    axes[0, 0].plot(self.xy[k, :, 0], self.xy[k, :, 1], color='blue')

        axes[0, 1].set_title('with phase function')

        for k in ix:
            axes[0, 1].plot(self.xy_phi[:, k, 0], self.xy_phi[:, k, 1], color = 'k')

        #for k in range(self.xy_phi.shape[0]):
        #    axes[0, 1].plot(self.xy_phi[k, :, 0], self.xy_phi[k, :, 1], color='blue')

        # let's look at speed as estimated by a differentiator
        dx = np.array(differentiator(self.xy[:, :, 0]))
        dy = np.array(differentiator(self.xy[:, :, 1]))

        speed = np.sqrt(dx**2 + dy**2)
        im = axes[1, 0].imshow(speed)
        fig.colorbar(im, ax = axes[1,0])

        dx = np.array(differentiator(self.xy_phi[:, :, 0]))
        dy = np.array(differentiator(self.xy_phi[:, :, 1]))

        speed_phi = np.sqrt(dx**2 + dy**2)

        im0 = axes[1,1].imshow(speed_phi)
        fig.colorbar(im0, ax = axes[1, 1])

        diff = np.diff(self.xy, axis = 1)
        dx = diff[:,:,0]
        dy = diff[:,:,1]

        diff = dx ** 2 + dy ** 2

        im1 = axes[2,0].imshow(diff)
        fig.colorbar(im1, ax = axes[2, 0])


        diff_phi = np.diff(self.xy_phi, axis=1)
        dx = diff_phi[:, :, 0]
        dy = diff_phi[:, :, 1]

        diff_phi = dx ** 2 + dy ** 2

        im2 = axes[2,1].imshow(diff_phi)
        fig.colorbar(im2,  ax = axes[2, 1])

        plt.show()
        plt.close()

        fig, axes = plt.subplots(ncols = 2)
        axes[0].imshow(np.abs(np.fft.fftshift(np.fft.fft2(diff - np.mean(diff)))))
        axes[1].imshow(np.abs(np.fft.fftshift(np.fft.fft2(diff_phi - np.mean(diff_phi)))))
        plt.show()
        plt.close()


        for k in range(speed_phi.shape[0]):
            fig, axes = plt.subplots(ncols = 2)

            Y = np.fft.fft(speed_phi[k])
            freq = np.fft.fftfreq(speed_phi.shape[1], self.t[1] - self.t[0])

            ix = np.where(freq > 0)

            axes[0].scatter(freq[ix]**-1, np.abs(Y[ix]))

            Y = np.fft.fft(speed[k])
            freq = np.fft.fftfreq(speed.shape[1], self.t[1] - self.t[0])

            ix = np.where(freq > 0)

            axes[0].scatter(freq[ix] ** -1, np.abs(Y[ix]))

            plt.show()
            plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(211, projection='3d')

        for k in ix:
            ax.plot(self.xy[:, k, 0], self.xy[:, k, 1], self.t, color='k')

        ax = fig.add_subplot(212, projection ='3d')
        for k in ix:
            ax.plot(self.xy_phi[:, k, 0], self.xy_phi[:, k, 1], self.t, color='k')

        plt.show()

        lim = max([self.alpha0, self.beta0])

        for k in range(diff.shape[1]):
            fig, axes = plt.subplots(ncols = 2)

            axes[0].plot(speed[:,k], color = 'k')
            axes[0].plot(speed_phi[:,k], color = 'b')

            print(simps(speed[:,k], np.array(range(0, speed.shape[0]))*2.7))
            print(simps(speed_phi[:, k], np.array(range(0, speed.shape[0])) * 2.7))

            axes[1].scatter(self.alpha0*np.cos(self.theta), self.beta0*np.sin(self.theta), color = 'k')
            axes[1].scatter([self.alpha0*np.cos(self.theta[k])], [self.beta0*np.sin(self.theta[k])], color = 'red')

            axes[1].plot(self.xy_phi[:,k,0], self.xy_phi[:,k,1], color = 'blue')
            axes[1].plot(self.xy[:, k, 0], self.xy[:, k, 1], color='blue')

            axes[1].set_xlim(-lim, lim)
            axes[1].set_ylim(-lim, lim)

            plt.show()





def solve_harmonic_series(x, theta, N = 3):
    A = []

    for t in theta:
        _ = []
        _.append(1.)
        
        for ix in range(1, N + 1):
            _.append(np.cos(t*ix))

        for ix in range(1, N + 1):
            _.append(np.sin(t*ix))


        #print _
        A.append(np.array(_))

    A = np.array(A)

    B = x.T

    co = np.linalg.lstsq(A, B)[0]

    return co

def compute_harmonic(theta, co, N = 3):
    A = []

    for t in theta:
        _ = []
        _.append(1.)
        
        for ix in range(1, N + 1):
            _.append(np.cos(t*ix))

        for ix in range(1, N + 1):
            _.append(np.sin(t*ix))


        #print _
        A.append(np.array(_))

    A = np.array(A)

    return A.dot(co)

def get_A(theta, N = 3):
    A = []

    for t in theta:
        _ = []
        _.append(1.)
        
        for ix in range(1, N + 1):
            _.append(np.cos(t*ix))

        for ix in range(1, N + 1):
            _.append(np.sin(t*ix))

        A.append(np.array(_))

    A = np.array(A)

    return A


def get_A_prime(theta, N=3):
    A = []

    for t in theta:
        _ = []
        _.append(0.)

        for ix in range(1, N + 1):
            _.append(-np.sin(t * ix) * ix)

        for ix in range(1, N + 1):
            _.append(np.cos(t * ix) * ix)

        A.append(np.array(_))

    A = np.array(A)

    return A

def get_C(accuracy = 2, n_out = 72):
    if accuracy == 2:
        c = np.array([-1., 1., 0.])
    elif accuracy == 4:
        c = np.array([1./12., -2./3., 0., 2./3., -1./12.])

    out = np.zeros((n_out, n_out, len(c)))

    for k in range(n_out):
        out_ = np.zeros((n_out, len(c)))
        out_[k] = c

        out[k] = out_[k]

    return out


