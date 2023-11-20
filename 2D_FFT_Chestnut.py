import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np


def normalize_complex_arr(a: np.ndarray) -> np.ndarray:
    # normalise to -1 to 1
    a_oo = a - a.real.min() - 1j * a.imag.min()
    return a_oo / np.abs(a_oo).max()


def get_2D_FFT(file_name, name):
    image_filename = file_name

    def calculate_2dft(input):
        ft = np.fft.ifftshift(input)
        ft = np.fft.fft2(ft)
        return np.fft.fftshift(ft)

    # Read and process image
    image = plt.imread(image_filename)
    image = image[::1, ::1, :3].mean(axis=2)

    ft = calculate_2dft(image)
    ft_real_01 = normalize_complex_arr(ft)
    ft_real_01 = normalize_complex_arr(np.clip(ft_real_01, 0, 0.015))

    print(f"{np.min(ft_real_01)=}")
    print(f"{np.max(ft_real_01)=}")

    plt.set_cmap("gray")
    # Show grayscale image and its Fourier transform

    plt.subplot(121)
    plt.imshow(image)
    plt.axis("off")
    plt.subplot(122)
    plt.title(name + ' FFT')
    plt.imshow(np.log(abs(ft_real_01)))
    plt.savefig("C:/Users/leele/PycharmProjects/FYP/Chestnut_Dec_FFTs/Chestnut_Dec_FFTs_" + name + '.png')
    # plt.axis("auto")
    plt.xlim(0, 160)
    plt.ylim(0, 160)
    plt.pause(2)


def GET_Dec_Chestnut_FFT(filename, foldername, channel):
    # Load the TIFF image
    image_path = filename
    image = Image.open(image_path)

    # Read Excel file containing boundary values
    bounds = "chestnut_nature_park_20201218_bounds.csv"
    df = pd.read_csv(bounds)

    # Extract boundary values
    for index, row in df.iterrows():
        x1, x2, y1, y2 = row['x0'], row['x1'], row['y0'], row['y1']
        name = row['name']

        # Crop the image using boundary values
        cropped_image = image.crop((x1, y1, x2, y2))
        saved_image = foldername + "/Chestnut_Dec_" + channel + "_" + name + '.png'
        cropped_image.save(saved_image)

        width, height = cropped_image.size

        if (width>=160 and height>=160):
            # Cropped square image size in px
            im_cropped = 160

            # Setting the points for cropped image (getting the middle 160x160 px crop)
            left = (width / 2) - (im_cropped / 2)
            top = (height / 2) - (im_cropped / 2)
            right = (width / 2) + (im_cropped / 2)
            bottom = (height / 2) + (im_cropped / 2)

            # Cropped image of above dimension
            # (It will not change original image)
            normalised_image = cropped_image.crop((left, top, right, bottom))
            saved_image_2 = foldername + "/Chestnut_Dec_" + channel + "_cropped_" + name + '.png'
            normalised_image.save(saved_image_2)

            # to test the equivariance of the 2D FFTs
            rotated_image = normalised_image.rotate(90)
            saved_image_3 = foldername + "/Chestnut_Dec_" + channel + "_cropped_rotated_" + name + '.png'
            rotated_image.save(saved_image_3)

            get_2D_FFT(saved_image_2, name)
            get_2D_FFT(saved_image_3, name)


GET_Dec_Chestnut_FFT("chestnut_nature_park_20201218_result.tif",
                     "C:/Users/leele/PycharmProjects/FYP/Chestnut_Dec_Green_2", "")


