import numpy as np

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
