import cv2
import numpy as np
import os
# P = 2M, Q = 2N


def _filter(shape, r, f):
    P, Q = shape[0], shape[1]
    x, y = np.meshgrid(np.arange(Q), np.arange(P))
    dist2 = (x-Q/2)**2 + (y-P/2)**2
    return f(dist2, r)
# hi-pass


def IHPF(dist2, r): return dist2 > r**2
def BPF(P, Q): pass
def GPF(P, Q): pass
# lo-pass


def part1(input_img_path, output_path):

    f = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    M, N = f.shape[0], f.shape[1]
    P, Q = 2*M, 2*N
    phi_P, phi_Q = phi(P), phi(Q)
    inv_phi_P, inv_phi_Q = phi_P.conj(), phi_Q.conj()

    F = FT(f, phi_P, phi_Q)

    H = _filter(F.shape, 100, IHPF)

    G = F*H

    f_processed = invFT(G, inv_phi_P, inv_phi_Q)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cv2.imwrite(os.path.join(output_path, "edges.png"), f_processed)
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
    M, N = img.shape[0], img.shape[1]
    M2, N2 = M//2, N//2
    return img[M2:M2+M, N2:N2+N]


def centerimg(img) -> np.array:
    M, N = img.shape[0], img.shape[1]
    P, Q = 2*M, 2*N
    x, y = np.meshgrid(np.arange(Q), np.arange(P))
    return img*(-1)**(x+y)


def phi(N):
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    return (1/(N**.5))*np.e**((-2*np.pi*x*y/N)*1j)


def FT(f: np.array, phi_P, phi_Q):

    f_p = add_pad(f.astype(float))
    f_p_c = centerimg(f_p)

    isGreyscale = len(f_p_c.shape) == 2
    f_p_c_t = np.transpose(
        f_p_c,
        (0, 1) if isGreyscale else (2, 0, 1)
    )

    return phi_P @ f_p_c_t @ phi_Q
    # np.transpose(fourier_centered_t, (0, 1) if isGreyscale else (1, 2, 0))


def invFT(F: np.array, inv_phi_P, inv_phi_Q):

    f = inv_phi_P @ F @ inv_phi_Q

    isGreyscale = len(f.shape) == 2

    f_t = np.transpose(
        f,
        (0, 1) if isGreyscale else (1, 2, 0)
    )
    f_t_r = f_t.real

    f_t_r_c = centerimg(f_t_r)

    return rm_pad(f_t_r_c)

def enhance_3 ( path_to_3 , output_path ) :
    img=cv2.imread( path_to_3)
    return img
    
def enhance_4 ( path_to_4 , output_path ) :
    img=cv2.imread( path_to_4)
    return img


def the2_write ( input_img_path , output_path ) :
    img=cv2.imread( input_img_path)
    img_name=""
    return img_name

def the2_read ( input_img_path ) :
    img=cv2.imread( input_img_path)
    cv2.imshow(img)
