import cv2
import numpy as np
import os
# P = 2M, Q = 2N


def filterfunc(shape, f, r, n):
    if len(shape) == 2:
        P, Q = shape[0], shape[1]
    else:
        P, Q = shape[1], shape[2]

    x, y = np.meshgrid(np.arange(Q), np.arange(P))
    dist2 = (x-Q/2)**2 + (y-P/2)**2
    return f(dist2, r**2, n)
# slope


def bslope(shape, angle, r, n):
    if len(shape) == 2:
        P, Q = shape[0], shape[1]
    else:
        P, Q = shape[1], shape[2]

    x, y = np.meshgrid(np.arange(Q), np.arange(P))

    m = np.tan(angle)

    Q2 = Q/2
    P2 = P/2
    dist2 = (-m*x+y+m*Q2-P2)**2/(m**2+1)
    return 1/(1+(dist2/(r**2))**n)


def gslope(shape, angle, r):
    if len(shape) == 2:
        P, Q = shape[0], shape[1]
    else:
        P, Q = shape[1], shape[2]

    x, y = np.meshgrid(np.arange(Q), np.arange(P))

    m = np.tan(angle)

    Q2 = Q/2
    P2 = P/2
    dist2 = (-m*x+y+m*Q2-P2)**2/(m**2+1)
    return np.e**-(dist2/(2*(r**2)))


def slope(shape, angle, r):
    if len(shape) == 2:
        P, Q = shape[0], shape[1]
    else:
        P, Q = shape[1], shape[2]

    x, y = np.meshgrid(np.arange(Q), np.arange(P))

    m = np.tan(angle)

    Q2 = Q/2
    P2 = P/2
    dist2 = (-m*x+y+m*Q2-P2)**2/(m**2+1)
    return dist2 < r**2
# laplacian
# def laplace(dist2, pi2, __): return 4*pi2 * dist2
# lo-pass


def ILPF(dist2, r2, _): return dist2 < r2
def BLPF(dist2, r2, n): return 1/(1+(dist2/r2)**n)
def GLPF(dist2, r2, _): return np.e**-(dist2/(2*r2))
# hi-pass
def IHPF(dist2, r2, _): return dist2 > r2
def BHPF(dist2, r2, n): return 1/(1+(r2/dist2)**n)
def GHPF(dist2, r2, _): return 1-np.e**-(dist2/(2*r2))
# # band-pass
# def ILPF(dist2, r2_min, r2_max, _): return dist2 > r2_min and dist2 < r2_max
# def BLPF(dist2, r2_min, r2_max, n): return 1/(1+(dist2/r2)**n)
# def GLPF(dist2, r2_min, r2_max, _): return np.e**-(dist2/(2*r2))
# # band-reject


def FT_img(input_path, output_path):
    f = cv2.imread(input_path)

    M, N = f.shape[0], f.shape[1]
    P, Q = 2*M, 2*N
    phi_P, phi_Q = phi(P), phi(Q)

    outbase = os.path.basename(output_path)
    tmp = outbase.split(".")
    output_path_base_noext = tmp[0]
    output_path_base_ext = ".".join(tmp[1:])

    F = FT(f, phi_P, phi_Q)

    F2 = (F.real**2 + F.imag**2)**.5

    B = F2[0, :, :]
    G = F2[1, :, :]
    R = F2[2, :, :]

    dir_r = output_path_base_noext + "_r." + output_path_base_ext
    dir_g = output_path_base_noext + "_g." + output_path_base_ext
    dir_b = output_path_base_noext + "_b." + output_path_base_ext

    outdir = os.path.dirname(output_path)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cv2.imwrite(os.path.join(outdir, dir_r), R)
    cv2.imwrite(os.path.join(outdir, dir_g), G)
    cv2.imwrite(os.path.join(outdir, dir_b), B)
    return F


def star(shape, r,):
    return \
        slope(shape, 0, r) +\
        slope(shape, np.pi/8, r) +\
        slope(shape, np.pi/4, r) +\
        slope(shape, 3*np.pi/8, r) +\
        slope(shape, np.pi/2, r) +\
        slope(shape, 5*np.pi/8, r) +\
        slope(shape, 3*np.pi/4, r) +\
        slope(shape, 7*np.pi/8, r)


def bstar(shape, r, n):
    return np.maximum.reduce([
        bslope(shape, 0, r, n),
        bslope(shape, np.pi/8, r, n),
        bslope(shape, np.pi/4, r, n),
        bslope(shape, 3*np.pi/8, r, n),
        bslope(shape, np.pi/2, r, n),
        bslope(shape, 5*np.pi/8, r, n),
        bslope(shape, 3*np.pi/4, r, n),
        bslope(shape, 7*np.pi/8, r, n)
    ])


def gstar(shape, r):
    return np.maximum.reduce([
        gslope(shape, 0, r),
        gslope(shape, np.pi/8, r),
        gslope(shape, np.pi/4, r),
        gslope(shape, 3*np.pi/8, r),
        gslope(shape, np.pi/2, r),
        gslope(shape, 5*np.pi/8, r),
        gslope(shape, 3*np.pi/4, r),
        gslope(shape, 7*np.pi/8, r)
    ])


def createDenoiseImg(denoiseImg, output_path):
    denoiseImg = denoiseImg.astype(float)

    denoiseImg -= np.min(denoiseImg)
    denoiseImg *= (255/np.max(denoiseImg))
    outdir = os.path.dirname(output_path)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    cv2.imwrite(output_path, denoiseImg)


def medianSpectrum(input_path, output_path):
    f = cv2.imread(input_path)

    M, N = f.shape[0], f.shape[1]
    P, Q = 2*M, 2*N
    phi_P, phi_Q = phi(P), phi(Q)
    inv_phi_P, inv_phi_Q = phi_P.conj(), phi_Q.conj()

    F = FT(f, phi_P, phi_Q)

    medF = cv2.medianBlur(F.real, 5) + cv2.medianBlur(F.imag, 5)*1j
    mask = (F - medF) > 10
    F[mask] = 0

    f_processed = invFT(F, inv_phi_P, inv_phi_Q)
    cv2.imwrite(output_path, f_processed)


def denoiseImg(f,
               #    denoiseImgr, denoiseImgg, denoiseImgb,
               _denoiseImg, output_path):

    M, N = f.shape[0], f.shape[1]
    P, Q = 2*M, 2*N
    phi_P, phi_Q = phi(P), phi(Q)
    inv_phi_P, inv_phi_Q = phi_P.conj(), phi_Q.conj()

    F = FT(f, phi_P, phi_Q)

    denoisedImg = F*_denoiseImg

    P, Q = denoisedImg.shape[1], denoisedImg.shape[2]

    f_processed = invFT(denoisedImg, inv_phi_P, inv_phi_Q)
    cv2.imwrite(output_path, f_processed)
    return f_processed


def part1(input_img_path, output_path):

    f = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    # f = cv2.imread(input_img_path)
    M, N = f.shape[0], f.shape[1]
    P, Q = 2*M, 2*N
    phi_P, phi_Q = phi(P), phi(Q)
    inv_phi_P, inv_phi_Q = phi_P.conj(), phi_Q.conj()

    F = FT(f, phi_P, phi_Q)

    H = filterfunc(F.shape, GHPF, 100, 2)

    G = F*H

    f_processed = invFT(G, inv_phi_P, inv_phi_Q)
    outdir = os.path.dirname(output_path)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    f_processed = np.where(f_processed < 0, 0, f_processed)
    f_processed = np.where(f_processed > 255, 255, f_processed)

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
    enhanced_img = denoiseImg(path_to_3, star((3036, 3000), 20),
                              "FT-Outputs/3.png")

    enhanced_img = np.uint8(enhanced_img)

    for _ in range(3):
        enhanced_img = cv2.medianBlur(enhanced_img, 5)
        enhanced_img = cv2.fastNlMeansDenoisingColored(
            enhanced_img, None, 10, 10, 7, 21)

    if output_path[-1] == "/":
        output_path.pop()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cv2.imwrite(output_path+"/enhanced.png", enhanced_img)
    return enhanced_img


def enhance_4(path_to_4, output_path):
    enhanced_img = cv2.imread(path_to_4)
    enhanced_img = cv2.medianBlur(enhanced_img, 5)
    enhanced_img = cv2.fastNlMeansDenoisingColored(
        enhanced_img, None, 10, 10, 7, 21)

    enhanced_img = cv2.filter2D(enhanced_img, -1,
                                np.array([
                                    [-1, -1, -1],
                                    [-1, 9, -1],
                                    [-1, -1, -1]]
                                ))

    if output_path[-1] == "/":
        output_path.pop()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cv2.imwrite(output_path+"/enhanced.png", enhanced_img)
    return enhanced_img


def the2_write(input_img_path, output_path):
    img = cv2.imread(input_img_path)
    img_name = ""
    return img_name


def the2_read(input_img_path):
    img = cv2.imread(input_img_path)
    cv2.imshow(img)
