import cv2
import numpy as np
from scipy import signal
import math
import matplotlib.pyplot as plt


def extract_frames(name_file):
    frames = []
    cap = cv2.VideoCapture(name_file)

    while cap.isOpened():
        succeeded, frame = cap.read()

        if succeeded:
            frames.append(frame.astype(dtype=np.int16))

        else:
            break

    cap.release()

    return frames


def calc_median(frames, curr_frame_index, k1):
    if curr_frame_index > k1:
        right = curr_frame_index + 1
        left = curr_frame_index - k1

    else:
        right = k1 + 1
        left = 0

    return np.median(frames[left:right], axis=0).astype(dtype=np.uint16)


def sign_background(frame, curr_median, threshold):
    rows, cols = frame.shape[0], frame.shape[1]
    new_frame = np.copy(frame)

    for row in range(rows):
        for col in range(cols):
            subtracted = abs(frame[row][col] - curr_median[row][col])
            if max(subtracted) < threshold:
                new_frame[row, col] = np.array([0, 0, 0])

    return new_frame


def median_change_dection(name_file, k1, k2, threshold):
    frames = extract_frames(name_file)
    curr_median = 0
    foreground_detected = []

    for i in range(len(frames)):
        if (i % k2) == 0:
            curr_median = calc_median(frames, i, k1)
        curr_frame_subtracted = sign_background(frames[i], curr_median, threshold)
        foreground_detected.append(curr_frame_subtracted)

    return foreground_detected


def count_foreground_objects(v_foreground, threshold):
    objects_counting = np.zeros(len(v_foreground))

    for i in range(len(v_foreground)):
        frame_binary_mask = tag_foregrounds(v_foreground[i], threshold)
        number_of_connected_comp, comp_map = cv2.connectedComponents(frame_binary_mask.astype('uint8'), connectivity=8)

        objects_counting[i] = number_of_connected_comp

    return objects_counting


def tag_foregrounds(frame, threshold):
    rows, cols = frame.shape[0], frame.shape[1]
    foreground_mask = np.zeros((rows, cols))

    for row in range(rows):
        for col in range(cols):

            if max(frame[row, col]) > threshold:
                foreground_mask[row, col] = 1

    return foreground_mask


def improve_foreground(v_foreground):
    frames = v_foreground
    improve_frames = []
    erosion_kernel = np.ones((7, 7), np.uint8)
    filter_kernel = np.ones((9, 9)) / 81

    for i in range(len(frames)):
        smoothed_img = cv2.filter2D(frames[i], -1, filter_kernel)

        frame_erosion = cv2.erode(smoothed_img, erosion_kernel, iterations=1)

        improve_frames.append(frame_erosion)

    save_output(improve_frames, "post_proc_change_detection.avi")
    return improve_frames


def save_output(v_foreground, out_file_name):
    height = v_foreground[0].shape[0]
    width = v_foreground[0].shape[1]

    out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width, height))

    for i in range(len(v_foreground)):
        out.write(v_foreground[i].astype(np.uint8))

    out.release()


def calc_mask_size(sigma):
    return math.ceil(2 * sigma)


def deriv_gauss_x(sigma, mask_size):
    x_coordinate = np.linspace(-mask_size, mask_size, mask_size * 2 + 1)
    y_coordinate = np.linspace(-mask_size, mask_size, mask_size * 2 + 1)

    columns_mask, rows_mask = np.meshgrid(x_coordinate, y_coordinate)

    # applying to each index in the matrix (-(x^2 + y^2)) the exp func
    exp_matrix = np.apply_along_axis(np.exp, 0, -(columns_mask * columns_mask + rows_mask * rows_mask) * (
            1 / (2 * sigma * sigma)))
    return ((-1 / (2 * math.pi * math.pow(sigma, 4))) * columns_mask) * exp_matrix


def deriv_gauss_y(sigma, mask_size):
    x_coordinate = np.linspace(-mask_size, mask_size, mask_size * 2 + 1)
    y_coordinate = np.linspace(-mask_size, mask_size, mask_size * 2 + 1)

    columns_mask, rows_mask = np.meshgrid(x_coordinate, y_coordinate)

    # applying to each index in the matrix (-(x^2 + y^2)) the exp func
    euler_matrix = np.apply_along_axis(np.exp, 0, -(rows_mask * rows_mask + columns_mask * columns_mask) * (
            1 / (2 * sigma * sigma)))
    return ((-1 / (2 * math.pi * math.pow(sigma, 4))) * rows_mask) * euler_matrix


def img_grad_by_axis(gradient, img):
    res = signal.convolve2d(img, gradient, boundary='symm', mode='same')

    return res


def gradient_magnitude(i_y, i_x):
    sum_of_squares_matrix = i_y * i_y + i_x * i_x
    magnitude_matrix = np.vectorize(math.sqrt)(sum_of_squares_matrix)  # apply sqrt for each pixel

    return magnitude_matrix


def smooth_img(img, sigma):
    mask_size = int(calc_mask_size(sigma))  # round up (2 * sigma)
    g_dx = deriv_gauss_x(sigma, mask_size)  # gaussian derivative with respect to x
    g_dy = deriv_gauss_y(sigma, mask_size)  # gaussian derivative with respect to y

    # applying the gradient
    i_x = img_grad_by_axis(g_dx, img)
    i_y = img_grad_by_axis(g_dy, img)

    img_mag = gradient_magnitude(i_x, i_y)

    return (i_x, i_y, img_mag)
    # return (i_x, i_y)


def extract_imgs_from_frames(name_file, f1_index, f2_index):
    frames = extract_frames(name_file)
    f1 = cv2.cvtColor(frames[f1_index].astype(dtype=np.uint16), cv2.COLOR_RGB2GRAY)
    f2 = cv2.cvtColor(frames[f2_index].astype(dtype=np.uint16), cv2.COLOR_RGB2GRAY)

    return f1, f2


def compute_gaussian_mask(sigma):
    mask_size = calc_mask_size(sigma)
    ax = np.linspace(-(mask_size), (mask_size), mask_size * 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    return kernel / np.sum(kernel)


def compute_window_vector(image, sigma, row, col):
    rows, cols = image.shape
    mask_size = calc_mask_size(sigma)
    window_offset = mask_size / 2
    lower_row = int(max((row - window_offset), 0))
    upper_row = int(min((row + window_offset + 1), rows))
    lower_col = int(max((col - window_offset), 0))
    upper_col = int(min((col + window_offset + 1), cols))

    neighbors_vector = (image[lower_row:upper_row, lower_col:upper_col]).flatten()

    return neighbors_vector


def compute_a(i_x, i_y, sigma_R, row, col):
    i_x_neighbors_vector = compute_window_vector(i_x, sigma_R, row, col)
    i_y_neighbors_vector = compute_window_vector(i_y, sigma_R, row, col)
    a_t = np.array([i_x_neighbors_vector, i_y_neighbors_vector])
    a = a_t.T

    return a


def compute_pseudo_inverse(i_x, i_y, sigma_R, row, col, height, width):
    a = compute_a(i_x, i_y, sigma_R, row, col)
    a_t = a.T
    pseudo_inverse = (np.linalg.inv(a_t.dot(a))).dot(a_t)

    return pseudo_inverse


def compute_c(g_i_x_square, g_i_y_square, g_i_x_y_y):
    c = np.zeros((2, 2))
    c[0, 0] = g_i_x_square
    c[0, 1] = g_i_x_y_y
    c[1, 0] = g_i_x_y_y
    c[1, 1] = g_i_y_square

    return c


def compute_optical_flow_by_gaussian(i_x, i_y, i_t, sigma_R, f1):
    rows, cols = f1.shape
    u, v = np.zeros((rows, cols)), np.zeros((rows, cols))
    gausian_mask = compute_gaussian_mask(sigma_R)
    i_x_square, i_y_square = i_x * i_x, i_y * i_y
    i_x_prod_i_y = i_x * i_y

    for row in range(rows):
        for col in range(cols):
            a = compute_a(i_x, i_y, sigma_R, row, col)
            c = a.T.dot(a)
            eigen_values, vec = np.linalg.eig(c)

            if (np.linalg.matrix_rank(c) == 2) and (min(eigen_values) > 0.001):
                pseudo_inverse = compute_pseudo_inverse(i_x, i_y, sigma_R, row, col, rows, cols)
                i_t_vector = -compute_window_vector(i_t, sigma_R, row, col)
                u[row, col] = pseudo_inverse.dot(i_t_vector)[0]
                v[row, col] = pseudo_inverse.dot(i_t_vector)[1]

    return u, v


def OF_plot_results(u, v, f1):
    width = u.shape[1]
    height = u.shape[0]
    x_coords = np.arange(0, width, 10)
    y_coords = np.arange(0, height, 10)

    X, Y = np.meshgrid(x_coords, y_coords)

    u = u[0:(height - 1):10, 0:(width - 1):10]
    v = v[0:(height - 1):10, 0:(width - 1):10]

    _fig, _ax = plt.subplots()

    _ax.imshow(f1, cmap='gray')

    _ax.quiver(X, Y, u, -v, width=0.005)

    _ax.xaxis.set_ticks([])

    _ax.yaxis.set_ticks([])

    plt.show()


def basic_LK_OF(name_file, nf1, nf2, sigma_S, sigma_R):
    f1, f2 = extract_imgs_from_frames(name_file, nf1, nf2)

    i_x, i_y, filtered_img1 = smooth_img(f1, sigma_S)

    i_t = np.subtract(f2.astype('float'), f1.astype('float'))
    u, v = compute_optical_flow_by_gaussian(i_x, i_y, i_t, sigma_R, f1)

    OF_plot_results(u, v, f1)

    return u, v, f1, f2


def affine_LK_OF(name_file, nf1, nf2, sigma_S, sigma_R):
    f1, f2 = extract_imgs_from_frames(name_file, nf1, nf2)
    i_x, i_y, filtered_img1 = smooth_img(f1, sigma_S)
    i_t = np.subtract(f2.astype('float'), f1.astype('float'))

    u, v = compute_affine_optical_flow(i_x, i_y, i_t, sigma_R, f1, 0.01)
    OF_plot_results(u, v, f1)

    return u, v, f1, f2


def calc_coefficient_vector(row, col, rows, cols, window_offset, coefficients_x_matrix, coefficients_y_matrix):
    lower_col = max(col - window_offset, 0)
    upper_col = min(cols, col + window_offset + 1)
    lower_row = max(row - window_offset, 0)
    upper_row = min(rows, row + window_offset + 1)

    x_coefficient = coefficients_x_matrix[lower_row:upper_row, lower_col:upper_col].flatten()
    y_coefficient = coefficients_y_matrix[lower_row:upper_row, lower_col:upper_col].flatten()

    return x_coefficient, y_coefficient


def calc_matrixes(rows, cols):
    x = np.linspace(0, cols - 1, cols)
    y = np.linspace(0, rows - 1, rows)

    x_matrix, y_matrix = np.meshgrid(x, y)

    return x_matrix, y_matrix


def compute_b(i_x_neighbors_vector, i_y_neighbors_vector, x_coefficient, y_coefficient):
    b = np.zeros((i_x_neighbors_vector.size, 6))

    for i in range(3):
        b[:, i] = i_x_neighbors_vector.T
        b[:, 3 + i] = i_y_neighbors_vector.T

    for row in range(i_x_neighbors_vector.size):
        b[row] = b[row] * (
            np.array([1, x_coefficient[row], y_coefficient[row], 1, x_coefficient[row], y_coefficient[row]]))

    return b


def compute_c_affine(i_x, i_y, sigma_R, row, col, coefficients_x_matrix, coefficients_y_matrix):
    window_offset = int(calc_mask_size(sigma_R) / 2)
    rows, cols = i_x.shape
    i_x_neighbors_vector = compute_window_vector(i_x, sigma_R, row, col)
    i_y_neighbors_vector = compute_window_vector(i_y, sigma_R, row, col)
    x_coefficient, y_coefficient = calc_coefficient_vector(row, col, rows, cols, window_offset, coefficients_x_matrix,
                                                           coefficients_y_matrix)
    b = compute_b(i_x_neighbors_vector, i_y_neighbors_vector, x_coefficient, y_coefficient)

    return b.T.dot(b), b.T


def compute_affine_optical_flow(i_x, i_y, i_t, sigma_R, f1, eigen_value_treshold):
    rows, cols = f1.shape
    u, v = np.zeros((rows, cols)), np.zeros((rows, cols))
    gausian_mask = compute_gaussian_mask(sigma_R)
    i_x = signal.convolve2d(i_x, gausian_mask, boundary='symm', mode='same')
    i_y = signal.convolve2d(i_y, gausian_mask, boundary='symm', mode='same')
    x_coefficient_matrix, y_coefficient_matrix = calc_matrixes(rows, cols)

    for row in range(rows):
        for col in range(cols):
            c, b_t = compute_c_affine(i_x, i_y, sigma_R, row, col, x_coefficient_matrix, y_coefficient_matrix)
            eigen_values, vec = np.linalg.eig(c)

            if (np.linalg.matrix_rank(c) == 6) and min(eigen_values) > eigen_value_treshold:
                pseudo_inverse = np.linalg.inv(c).dot(b_t)
                i_t_neighbors_vector = compute_window_vector(i_t, sigma_R, row, col)
                coefficients = pseudo_inverse.dot(-i_t_neighbors_vector)
                u[row, col] = coefficients[0:3].dot(np.array([1, col, row]))
                v[row, col] = coefficients[3:6].dot(np.array([1, col, row]))

    return u, v


def create_flow_matrix(u, v):
    rows, cols = u.shape
    flow = np.zeros((rows, cols, 2))

    for row in range(rows):
        for col in range(cols):
            flow[row][col] = np.array([u[row, col], v[row, col]])

    return flow


def eval_OF(im1, im2, u, v):
    w_im1 = warp_flow(im1, u, v).astype('int')
    w_diff = abs(im2.astype('int') - w_im1)
    err = np.sum(w_diff ** 2)

    return w_im1, w_diff, err


def warp_flow(img, u, v):
    flow = create_flow_matrix(u, v).astype('float32')
    h, w = flow.shape[:2]
    new_flow = -flow
    new_flow[:, :, 0] += np.arange(w)
    new_flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, new_flow, None, cv2.INTER_LINEAR)
    return res


if __name__ == '__main__':
    v_foreground = median_change_dection("SLIDE.avi", 20, 15, 30)
    post_proc_frames = improve_foreground(v_foreground)
    counting_objects = count_foreground_objects(v_foreground, 25)
    u, v, f1, f2 = basic_LK_OF("SLIDE.avi", 60, 61, 2, 30)
    eval_OF(f1, f2, u, v)

    u, v, f1, f2 = affine_LK_OF("sphere.gif", 4, 5, 0.8, 10)
    eval_OF(f1, f2, u, v)
