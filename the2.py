import cv2
import numpy as np
import os
# P = 2M, Q = 2N


def _filter(shape, r, n, f):
    if len(shape) == 2:
        P, Q = shape[0], shape[1]
    else:
        P, Q = shape[1], shape[2]

    x, y = np.meshgrid(np.arange(Q), np.arange(P))
    dist2 = (x-Q/2)**2 + (y-P/2)**2
    return f(dist2, r**2, n)
# lo-pass
def ILPF(dist2, r2, _): return dist2 < r2
def BLPF(dist2, r2, n): return 1/(1+(dist2/r2)**n)
def GLPF(dist2, r2, _): return np.e**-(dist2/(2*r2))
# hi-pass
def IHPF(dist2, r2, _): return dist2 > r2
def BHPF(dist2, r2, n): return 1/(1+(r2/dist2)**n)
def GHPF(dist2, r2, _): return 1-np.e**-(dist2/(2*r2))


def part1(input_img_path, output_path):

    f = cv2.imread(input_img_path)
    M, N = f.shape[0], f.shape[1]
    P, Q = 2*M, 2*N
    phi_P, phi_Q = phi(P), phi(Q)
    inv_phi_P, inv_phi_Q = phi_P.conj(), phi_Q.conj()

    F = FT(f, phi_P, phi_Q)

    H = _filter(F.shape, 300, 2, IHPF)

    G = F*H

    f_processed = invFT(G, inv_phi_P, inv_phi_Q)
    outdir = os.path.dirname(output_path)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cv2.imwrite(output_path, f_processed)
    return f_processed


def add_pad(img) -> np.array:
    M, N = img.shape[0], img.shape[1]
    P, Q = 2*M, 2*N
    M2, N2 = M//2, N//2

    newshape = list(img.shape)
    newshape[0] = P
    newshape[1] = Q

    padded_img = np.zeros(tuple(newshape))

    padded_img[M2:M2+M, N2:N2+N] = img

    return padded_img


def rm_pad(img) -> np.array:
    P, Q = img.shape[0], img.shape[1]
    M, N = P//2, Q//2
    M2, N2 = M//2, N//2
    return img[M2:M2+M, N2:N2+N]


def centerimg(img, P, Q) -> np.array:
    x, y = np.meshgrid(np.arange(Q), np.arange(P))
    return img*(-1)**(x+y)


def phi(N):
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    return (1/(N**.5))*np.e**((-2*np.pi*x*y/N)*1j)


def FT(f: np.array, phi_P, phi_Q):

    f_p = add_pad(f.astype(float))
    P, Q = f_p.shape[0], f_p.shape[1]
    f_p_t = np.transpose(
        f_p,
        (0, 1) if len(f_p.shape) == 2 else (2, 0, 1)
    )
    f_p_t_c = centerimg(f_p_t, P, Q)

    return phi_P @ f_p_t_c @ phi_Q
    # np.transpose(fourier_centered_t, (0, 1) if isGreyscale else (1, 2, 0))


def invFT(F: np.array, inv_phi_P, inv_phi_Q):

    f = inv_phi_P @ F @ inv_phi_Q

    f_r = f.real

    if len(f_r.shape) == 2:
        P, Q = f_r.shape[0], f_r.shape[1]
        axes = (0, 1)
    else:
        P, Q = f_r.shape[1], f_r.shape[2]
        axes = (1, 2, 0)

    f_r_c = centerimg(f_r, P, Q)

    f_r_c_t = np.transpose(
        f_r_c,
        axes
    )

    return rm_pad(f_r_c_t)


def enhance_3(path_to_3, output_path):
    img = cv2.imread(path_to_3)
    return img


def enhance_4(path_to_4, output_path):
    img = cv2.imread(path_to_4)
    return img


def the2_write(input_img_path, output_path):
    img = cv2.imread(input_img_path)
    img_name = ""
    return img_name


def the2_read(input_img_path):
    img = cv2.imread(input_img_path)
    cv2.imshow(img)
