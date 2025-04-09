import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift


def generate_psf(wavelength:float=0.5e-6, f:list[float]=0.1, D:float= 0.01,delta_z:list[float]=[0, 0.001, 0.005],N:int=512) -> np.ndarray:
    """
    parameter setting:
    wavelength: imaging wave length (unit: meter)
    f:          focus (unit: meter)
    D:          aperature diameter (unit:meter)
    delta_z:    aperature diameter (unit:meter)
    N:          account of sample point
    """
    # generate coordinate
    x = np.linspace(-D/2, D/2, N)
    y = np.linspace(-D/2, D/2, N)
    X, Y = np.meshgrid(x, y)

    # pupil function
    pupil = np.sqrt(X**2 + Y**2) <= D/2

    # phase error by defocus
    k = 2 * np.pi / wavelength
    psfs = np.zeros((N, N, len(delta_z)), dtype=np.float32)
    for i, dz in enumerate(delta_z):
        W20 = dz * (D/2)**2 / (2 * f**2)  # wavefront error by defocus
        phase_defocus = np.exp(1j * k * W20 * (X**2 + Y**2) / (f**2))

        # pupil by defocus
        pupil_defocus = pupil * phase_defocus

        # calculate PSF
        U = fftshift(fft2(fftshift(pupil_defocus)))
        psfs[:,:,i] = np.abs(U)**2
        psfs[:,:,i] = psfs[:,:,i] / psfs[:,:,i].sum()  # normalize

    return pupil, psfs

if __name__ == '__main__':

    # defocus
    delta_z = [0, 0.001, 0.005, 0.01, 0.5]

    pupil, psfs = generate_psf(delta_z=delta_z)

    # vision
    plt.figure(figsize=(10, 5))
    plt.subplot(2,3,1), plt.imshow(pupil, cmap='gray'), plt.title('Pupil Function')
    for i in range(psfs.shape[2]):
        plt.subplot(2,3,i+2), plt.imshow(np.log(psfs[:,:,i] + 1e-6), cmap='jet'), plt.title('Defocused PSF (Log Scale)')

    plt.tight_layout()
    plt.show()